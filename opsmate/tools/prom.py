from opsmate.dino.types import ToolCall, PresentationMixin
from pydantic import Field, computed_field, PrivateAttr
from typing import Any, List, Dict, ClassVar, Optional
from httpx import AsyncClient
from opsmate.dino import dino
from opsmate.dino.types import Message
from opsmate.tools.datetime import DatetimeRange, datetime_extraction
from opsmate.tools.knowledge_retrieval import KnowledgeRetrieval
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotext
from datetime import datetime
import base64
from functools import lru_cache
from asyncio import Semaphore, create_task, gather
import structlog
from uuid import uuid4
from opsmate.dbq.dbq import dbq_task, enqueue_task
import json
from datetime import UTC, timedelta
import random
from sqlmodel import Session
from opsmate.knowledgestore.models import Category, aconn
from copy import deepcopy
import time
import os
import io
from tenacity import retry, stop_after_attempt, wait_fixed

logger = structlog.get_logger(__name__)

DEFAULT_ENDPOINT = "http://localhost:9090"
DEFAULT_PATH = "/api/v1/query_range"


def backoff_func(retry_count: int):
    return datetime.now(UTC) + timedelta(
        milliseconds=2 ** (retry_count - 1) + random.uniform(0, 10)
    )


@dbq_task(
    retry_on=(Exception,),
    max_retries=10,
    back_off_func=backoff_func,
)
async def ingest_metrics(metrics: List[Dict[str, Any]], prom_endpoint: str):
    kbs = []
    for metric in metrics:
        kbs.append(
            {
                "uuid": str(uuid4()),
                "id": 1,  # doesn't matter
                # "summary": chunk.metadata["summary"],
                "categories": [
                    Category.OBSERVABILITY.value,
                    Category.PROMETHEUS.value,
                ],
                "data_source_provider": "prometheus",
                "data_source": prom_endpoint,
                "metadata": json.dumps({"metric": metric["metric_name"]}),
                "path": metric["metric_name"],
                "content": json.dumps(metric),
                "created_at": datetime.now(),
            }
        )
    conn = await aconn()
    table = await conn.open_table("knowledge_store")
    await table.merge_insert(
        on=["path", "data_source", "data_source_provider"]
    ).when_matched_update_all().when_not_matched_insert_all().execute(kbs)

    await table.optimize()


class PromQL:
    DEFAULT_LABEL_BLACKLIST: tuple[str, ...] = (
        "pod",
        "container",
        "container_id",
        "endpoint",
        "uuid",
        "uid",
        "id",
        "instance",
        "image",
        "name",
        "mountpoint",
        "device",
    )

    def __init__(
        self,
        endpoint: str,
        user_id: str | None = None,
        api_key: str | None = None,
        client: AsyncClient = AsyncClient(),
    ):
        self.endpoint = endpoint
        self.user_id = user_id
        self.api_key = api_key
        self.client = client

    async def load_metrics(
        self,
        force_reload=False,
        with_labels=True,
        label_blacklist=DEFAULT_LABEL_BLACKLIST,
    ):
        if force_reload or not hasattr(self, "df"):
            self.metrics = await self.fetch_metrics(
                with_labels=with_labels, label_blacklist=label_blacklist
            )
            return self.metrics

        return self.metrics

    async def fetch_metrics(
        self, with_labels=True, label_blacklist=DEFAULT_LABEL_BLACKLIST
    ):
        response = await self.client.get(
            self.endpoint + "/api/v1/label/__name__/values", headers=self.headers()
        )
        response_json = response.json()

        if response.status_code != 200:
            logger.error(
                "Failed to fetch metrics from prometheus", response_json=response_json
            )
            raise Exception("Failed to fetch metrics from prometheus")

        metrics_data = response_json["data"]

        metrics = []
        for metric in metrics_data:
            metrics.append({"metric_name": metric})

        semaphore = Semaphore(10)

        async def _apply_labels(metric):
            async with semaphore:
                metric["labels"] = await self.get_metric_labels(
                    metric["metric_name"], label_blacklist=label_blacklist
                )
                return metric

        tasks = []
        for metric in metrics:
            tasks.append(create_task(_apply_labels(metric)))

        await gather(*tasks)

        return metrics

    async def ingest_metrics(self, session: Session):
        await self.load_metrics()

        for i in range(0, len(self.metrics), 20):
            start = time.time()
            logger.info("ingesting metrics", batch_id=i, len=len(self.metrics))
            # await ingest_metrics(self.metrics[i : i + 20], self.endpoint)
            enqueue_task(
                session,
                ingest_metrics,
                self.metrics[i : i + 20],
                self.endpoint,
                queue_name="lancedb-batch-ingest",
            )
            end = time.time()
            logger.info(
                "ingested metrics",
                batch_id=i,
                len=len(self.metrics),
                time=f"{end - start:.2f}s",
            )

    @lru_cache
    async def get_metric_labels(
        self, metric_name, label_blacklist=DEFAULT_LABEL_BLACKLIST
    ):
        response = await self.client.get(
            self.endpoint + "/api/v1/labels",
            params={"match[]": metric_name},
            headers=self.headers(),
        )
        response_json = response.json()
        if response.status_code != 200:
            logger.error(
                "Failed to fetch labels for metric: " + metric_name,
                response_json=response_json,
            )
            return []
        labels = response_json["data"]
        labels.remove("__name__") if "__name__" in labels else None

        result = {}
        for label in labels:
            if label in label_blacklist:
                result[label] = ""
                continue
            response = await self.client.get(
                self.endpoint + "/api/v1/label/" + label + "/values",
                params={"match[]": metric_name},
                headers=self.headers(),
            )
            response_json = response.json()
            if response.status_code != 200:
                logger.error(
                    "Failed to fetch values for label: " + label,
                    response_json=response_json,
                )
                result[label] = ""
                continue
            values = response_json["data"]
            # result[label] = str.join("|", values)
            joint_values = "|".join(values)
            if len(joint_values) > 100:
                result[label] = joint_values[:100] + "..."
            else:
                result[label] = joint_values

        return result

    def headers(self):
        h = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "opsmate prometheus tool",
        }
        if self.user_id and self.api_key:
            b64_token = base64.b64encode(
                f"{self.user_id}:{self.api_key}".encode()
            ).decode()
            h["Authorization"] = f"Basic {b64_token}"
        return h


class PromQuery(ToolCall[dict[str, Any]], DatetimeRange, PresentationMixin):
    """
    A tool to query metrics from Prometheus
    """

    query: str = Field(description="The prometheus query")

    y_label: str = Field(
        description="The y-axis label of the time series based on the query",
        default="Value",
    )
    x_label: str = Field(
        description="The x-axis label of the time series based on the query",
        default="Timestamp",
    )
    title: str = Field(
        description="The title of the time series based on the query",
        default="Time Series Data",
    )
    explanation: str = Field(
        description="A brief explanation of the query",
    )

    data_points_per_series: int = Field(
        description="The number of points per time series",
        default=100,
        le=200,
        ge=100,
    )
    sample_points: ClassVar[int] = 20

    @computed_field
    def step(self) -> str:
        # no more than 10,000 points
        secs = (
            self.end_dt - self.start_dt
        ).total_seconds() / self.data_points_per_series
        if secs < 1:
            return "15s"
        else:
            return f"{int(secs)}s"

    def headers(self, context: dict[str, Any] = {}):
        h = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "opsmate prometheus tool",
        }
        if context.get("prometheus_api_key"):
            user_id = context.get("prometheus_user_id")
            token = context.get("prometheus_api_key")
            b64_token = base64.b64encode(f"{user_id}:{token}".encode()).decode()
            h["Authorization"] = f"Basic {b64_token}"
        return h

    async def __call__(self, context: dict[str, Any] = {}):
        # TBD wheather we want to make the envvar injection the default behaviour
        context = context.copy()
        envvars = os.environ.copy()
        context.update(envvars)
        endpoint = context.get("prometheus_endpoint", DEFAULT_ENDPOINT)
        path = context.get("prometheus_path", DEFAULT_PATH)

        response = await AsyncClient().post(
            endpoint + path,
            data={
                "query": self.query,
                "start": self.start,
                "end": self.end,
                "step": self.step,
            },
            headers=self.headers(context),
        )

        if response.status_code != 200:
            logger.error(
                "Failed to fetch metrics from prometheus", response_json=response.json()
            )
            return {
                "error": "Failed to fetch metrics from prometheus",
                "data": response.text,
            }

        return response.json()

    _dataframe: pd.DataFrame | None = PrivateAttr(default=None)

    @property
    def dataframe(self):
        if self._dataframe is not None:
            return self._dataframe

        datapoints = deepcopy(self.output["data"]["result"])

        if len(datapoints) == 0:
            logger.warning("No datapoints found for query", query=self.query)
            return None

        # Create DataFrame from all results
        df_list: list[pd.DataFrame] = []
        for result in datapoints:
            metric_name = "-".join(result["metric"].values())
            # Convert values to float during DataFrame creation
            df = pd.DataFrame(result["values"], columns=["timestamp", metric_name])
            df[metric_name] = df[metric_name].astype(float)  # Ensure values are float
            df_list.append(df)

        # Merge all dataframes on timestamp
        if len(df_list) == 1:
            df = df_list[0]
        else:
            # Progressively merge all dataframes
            df = df_list[0]
            for additional_df in df_list[1:]:
                df = df.merge(additional_df, on="timestamp", how="outer")

        df = df.fillna(0.0)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        self._dataframe = df
        return df

    def sampled_output(self):
        if "error" in self.output:
            return self.output

        output = deepcopy(self.output)
        results = []
        for result in output["data"]["result"]:
            values = result["values"]
            metric = result["metric"]

            # Aim to sample around 10 points
            if len(values) > self.sample_points:
                step = max(1, len(values) // self.sample_points)
                values = values[::step]

            results.append(
                {
                    "metric": metric,
                    "values": values,
                }
            )
        output["data"]["result"] = results
        return output

    def markdown(self, context: dict[str, Any] = {}): ...

    def time_series(self, in_terminal: bool = False, show_base64_image: bool = False):
        if "error" in self.output:
            return

        logger.info("plotting time series", query=self.query)
        plt.figure(figsize=(12, 6))

        for result in self.output["data"]["result"]:
            values = result["values"]
            metric = result["metric"]
            metric_name = "-".join(metric.values())
            timestamps = [datetime.fromtimestamp(ts) for ts, _ in values]
            measurements = [float(val) for _, val in values]
            df = pd.DataFrame({"timestamp": timestamps, "measurement": measurements})
            if in_terminal:
                df["timestamp"] = mdates.date2num(df["timestamp"])
            plt.plot(df["timestamp"], df["measurement"], label=metric_name)
        plt.grid(True)
        plt.title(f"{self.title} - {self.query}")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
        plt.tight_layout()

        if in_terminal:
            plt.show(block=False)
            fig = plt.gcf()
            plotext.from_matplotlib(fig)
            plotext.show()
        if show_base64_image:
            # get the base64 image
            image_data = io.BytesIO()
            plt.savefig(image_data, format="png")
            image_data.seek(0)
            return {
                "title": self.title,
                "mime_type": "image/png",
                "data": base64.b64encode(image_data.read()).decode(),
            }
        else:
            plt.show()


@dino(
    model="claude-3-7-sonnet-20250219",
    response_model=PromQuery,
    tools=[datetime_extraction, KnowledgeRetrieval],
)
async def prometheus_query(query: str, context: dict[str, Any] = {}):
    """
    You are a world class SRE who excels at querying metrics from Prometheus
    You are given a query in natural language and you need to convert it into a valid Prometheus query
    Please think carefully and generate 3 different PromQL queries, and then choose the best one among them as the answer.

    <tasks>
    - Parse the natural language query
    - Genereate a PromQL query that fulfills the request
    - Provide a brief explanation of the query
    </tasks>

    <important>
    - use `datetime_extraction` tool to get the time range of the query
    - use `KnowledgeRetrieval` tool to get the metrics and labels that are relevant to the query
    - DO NOT use labels that are not present in the metrics based on the knowledge retrieval. e.g. by(label_not_in_metrics) or metric{label_not_in_metrics="xxx"} should be avoided.
    - DO NOT use metrics that do not exist from knowledge retrieval.
    - USE `_bucket` suffix metrics if the query is about histograms.
    - The rate interval must be greater or equal to 2m.
    - Avoid using `avg` unless you are told to, as it's statistically meaningless.
    - use `sum(rate(...))` for rate related queries and avoid use rate(...) directly.
    - When you use regex to match labels, use `=~"(.*)THE_PATTERN(.*)"` to match the pattern.
    </important>
    """

    return [
        Message.user(query),
    ]


class PrometheusTool(ToolCall[PromQuery], PresentationMixin):
    """
    PrometheusTool is a tool to query metrics from prometheus tsdb via natural language
    """

    natural_language_query: str = Field(
        description="The natural language query to be translated into a prometheus query"
    )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def __call__(self, context: dict[str, Any] = {}):
        context = context.copy()
        context["llm_summary"] = False
        if "top_n" not in context:
            context["top_n"] = 20
        model = context.get("dino_model", "claude-3-7-sonnet-20250219")

        prom_query: PromQuery = await prometheus_query(
            self.natural_language_query,
            context=context,
            model=model,
        )

        await prom_query.run(context)
        if "error" in prom_query:
            raise Exception(prom_query["error"])
        return prom_query

    def markdown(self, context: dict[str, Any] = {}):
        # XXX: default in_terminal to False
        in_terminal = context.get("in_terminal", False)
        if in_terminal:
            self.time_series(in_terminal)

        return f"""
### Natural Language Query

{self.natural_language_query}

### Prometheus Query

```
{self.output.query}
```

### Explanation

{self.output.explanation}
"""

    def time_series(self, in_terminal: bool = False, show_base64_image: bool = False):
        return self.output.time_series(
            in_terminal=in_terminal, show_base64_image=show_base64_image
        )

    def prompt_display(self):
        m = self.model_dump()
        m["output"]["output"] = self.output.sampled_output()
        return json.dumps(m, default=str)
