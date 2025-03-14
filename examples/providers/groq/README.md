# Groq Provider

This example demonstrates how to register a new provider. In this case we are registering [Groq](https://groq.com) as the LLM provider for OpsMate.

## Installation

```bash
pip install -e .
```

After installation you can list all the models via

```bash
$ opsmate list-models
                   Models
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Provider  ┃ Model                         ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ openai    │ gpt-4o                        │
├───────────┼───────────────────────────────┤
│ openai    │ gpt-4o-mini                   │
├───────────┼───────────────────────────────┤
│ openai    │ o1-preview                    │
├───────────┼───────────────────────────────┤
│ anthropic │ claude-3-5-sonnet-20241022    │
├───────────┼───────────────────────────────┤
│ anthropic │ claude-3-7-sonnet-20250219    │
├───────────┼───────────────────────────────┤
│ xai       │ grok-2-1212                   │
├───────────┼───────────────────────────────┤
│ groq      │ qwen-2.5-32b                  │
├───────────┼───────────────────────────────┤
│ groq      │ deepseek-r1-distill-qwen-32b  │
├───────────┼───────────────────────────────┤
│ groq      │ deepseek-r1-distill-llama-70b │
├───────────┼───────────────────────────────┤
│ groq      │ llama-3.3-70b-versatile       │
└───────────┴───────────────────────────────┘
```

You will notice that the models from Groq are automatically added to the list of models.

You can use the `-m` flag to specify the model to use. For example:

```bash
export OPSMATE_LOGLEVEL=ERROR
$ opsmate run -n --tools HtmlToText -m llama-3.3-70b-versatile "find me top 10 news on the hacker news, titl
e only in bullet points"
The top 10 news on Hacker News are:
* The most unhinged video wall, made out of Chromebooks
* Show HN: Berlin Swapfest – Electronics flea market
* GLP-1 drugs – the biggest economic disruptor since the internet? (2024)
* Efabless – Shutdown Notice
* Video encoding requires using your eyes
* Making o1, o3, and Sonnet 3.7 hallucinate for everyone
* How to gain code execution on hundreds of millions of people and popular apps
* Show HN: I made a website where you can create your own "Life in Weeks" timeline
* Drone captures narwhals using their tusks to explore, forage and play
* Maestro – Next generation mobile UI automation
```
