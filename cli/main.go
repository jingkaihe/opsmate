package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	opsmatesdk "github.com/jingkaihe/opsmate/cli/sdk"
	"github.com/olekukonko/tablewriter"
	"github.com/urfave/cli/v2"
)

type clientCtxKey struct{}

func main() {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}

	app := &cli.App{
		Name:  "opsmate",
		Usage: "opsmate is a command line tool for solving production issues",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:  "endpoint",
				Usage: "endpoint to use",
				Value: "http://127.0.0.1:8000",
			},
			&cli.StringFlag{
				Name:  "inventory-dir",
				Usage: "directory to store inventory",
				Value: filepath.Join(homeDir, ".config", "opsmate"),
			},
			&cli.DurationFlag{
				Name:  "inventory-ttl",
				Usage: "how long the inventory is valid",
				Value: 10 * time.Minute,
			},
			&cli.StringFlag{
				Name:  "provider",
				Usage: "provider to use",
				Value: "openai",
			},
			&cli.StringFlag{
				Name:  "model",
				Usage: "model to use",
				Value: "gpt-4o",
			},
		},
		Before: func(c *cli.Context) error {
			client, err := getClient(c.String("endpoint"))
			if err != nil {
				return err
			}
			c.Context = context.WithValue(c.Context, clientCtxKey{}, client)

			if err := RecordInventory(c.String("inventory-dir"), c.Duration("inventory-ttl")); err != nil {
				return err
			}
			return nil
		},
		Commands: []*cli.Command{
			{
				Name:  "list-models",
				Usage: "list available large language models",
				Action: func(c *cli.Context) error {
					client := c.Context.Value(clientCtxKey{}).(*opsmatesdk.APIClient)

					req := client.DefaultAPI.ModelsApiV1ModelsGet(c.Context)

					models, resp, err := req.Execute()
					if err != nil {
						return err
					}
					defer resp.Body.Close()

					table := tablewriter.NewWriter(os.Stdout)
					table.SetHeader([]string{"PROVIDER", "MODEL"})

					for _, model := range models {
						table.Append([]string{
							model.GetProvider(),
							model.GetModel(),
						})
					}

					table.Render()
					return nil
				},
			},
			{
				Name:  "status",
				Usage: "get the status of the server",
				Action: func(c *cli.Context) error {
					client := c.Context.Value(clientCtxKey{}).(*opsmatesdk.APIClient)
					req := client.DefaultAPI.HealthApiV1HealthGet(c.Context)
					health, resp, err := req.Execute()
					if err != nil {
						return err
					}
					defer resp.Body.Close()
					fmt.Println(health.GetStatus())
					return nil
				},
			},
			{
				Name:        "run",
				Usage:       "opsmate run <instruction>",
				Description: "execute commands based on the instruction in natural language",
				Flags: []cli.Flag{
					&cli.StringSliceFlag{
						Name:  "contexts",
						Usage: "contexts to use",
						Value: cli.NewStringSlice("cli"),
					},
				},
				Args: true,
				Action: func(c *cli.Context) error {
					var (
						client      = c.Context.Value(clientCtxKey{}).(*opsmatesdk.APIClient)
						provider    = c.String("provider")
						model       = c.String("model")
						contexts    = c.StringSlice("contexts")
						instruction = c.Args().First()
					)

					req := client.DefaultAPI.RunApiV1RunPost(c.Context)
					req = req.RunRequest(opsmatesdk.RunRequest{
						Provider:    provider,
						Model:       model,
						Instruction: instruction,
						Contexts:    contexts,
					})
					results, resp, err := req.Execute()
					if err != nil {
						return err
					}
					defer resp.Body.Close()

					for _, result := range results {
						table := tablewriter.NewWriter(os.Stdout)
						// Extract keys and values from the map
						keys := make([]string, 0, len(result))
						values := make([]string, 0, len(result))
						for k, v := range result {
							keys = append(keys, k)
							values = append(values, fmt.Sprintf("%v", v))
						}
						table.SetHeader(keys)
						table.Append(values)
						table.Render()
					}
					return nil
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}

func getClient(endpoint string) (*opsmatesdk.APIClient, error) {
	cfg := opsmatesdk.NewConfiguration()

	url, err := url.Parse(endpoint)
	if err != nil {
		return nil, err
	}
	cfg.Host = url.Host
	cfg.Scheme = url.Scheme
	client := opsmatesdk.NewAPIClient(cfg)
	return client, nil
}

func RecordInventory(inventoryDir string, inventoryTTL time.Duration) error {
	inventory, err := osInventory()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(inventoryDir, 0755); err != nil {
		return err
	}

	filename := filepath.Join(inventoryDir, "inventory.json")
	stats, err := os.Stat(filename)
	if err == nil && time.Since(stats.ModTime()) < inventoryTTL {
		return nil
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(inventory)
}

// osInventory returns a map of os inventory information
func osInventory() (map[string]interface{}, error) {
	memInGB, err := getMemInfo()
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"os":      runtime.GOOS,
		"arch":    runtime.GOARCH,
		"cpus":    runtime.NumCPU(),
		"memInGB": memInGB,
	}, nil
}

func getMemInfo() (float64, error) {
	var info syscall.Sysinfo_t
	err := syscall.Sysinfo(&info)
	if err != nil {
		return 0, err
	}

	totalRam := info.Totalram * uint64(info.Unit) // Total RAM in bytes
	memInGB := float64(totalRam) / (1024 * 1024 * 1024)
	// round to 2 decimal places
	return math.Round(memInGB*100) / 100, nil
}
