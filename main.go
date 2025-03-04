// Copyright 2025 The Box Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/alixaxel/pagerank"
)

// Prompt is a llm prompt
type Prompt struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// Query submits a query to the llm
func Query(query string) string {
	prompt := Prompt{
		Model:  "llama3.2",
		Prompt: query,
	}
	data, err := json.Marshal(prompt)
	if err != nil {
		panic(err)
	}
	buffer := bytes.NewBuffer(data)
	response, err := http.Post("http://10.0.0.54:11434/api/generate", "application/json", buffer)
	if err != nil {
		panic(err)
	}
	reader, answer := bufio.NewReader(response.Body), ""
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		data := map[string]interface{}{}
		err = json.Unmarshal([]byte(line), &data)
		text := data["response"].(string)
		answer += text
	}
	return answer
}

// PageRank uses page rank to turn the output of attention into a distribution
func PageRank(a Matrix) []float32 {
	a = a.T()
	z := make([]float32, a.Rows)
	graph := pagerank.NewGraph()
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Rows; j++ {
			cs := NCS(a.Data[i*a.Cols:(i+1)*a.Cols], a.Data[j*a.Cols:(j+1)*a.Cols])
			graph.Link(uint32(i), uint32(j), float64(cs))
		}
	}
	graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
		z[node] = float32(rank)
	})
	return z
}

func main() {
	type Vector struct {
		Vector []float32
		Symbol byte
	}
	mind, index := make([]Vector, 128*1024), 0
	m := NewMixer()
	m.Add(0)
	query := "Why is time symmetric at quantum scales by asymmetric at large scales?"
	for {
		answer := Query(query)
		if len(answer) != 0 {
			fmt.Printf(answer)
			fmt.Println("\n----------------------------------------")
			for _, v := range []byte(answer) {
				index = (index + 1) % len(mind)
				vectors := m.Mix()
				//vector := PageRank(vectors)
				mind[index].Vector = vectors.Data
				mind[index].Symbol = v
				m.Add(v)
			}
		}
		fmt.Println("half way there")
		query = ""
		i := 0
		for {
			q := m.Mix()
			//qq := PageRank(q)
			max, symbol := float32(0.0), byte(0)
			/*for i, v := range qq {
				if v > max {
					max, symbol = v, byte(i)
				}
			}*/
			for _, v := range mind {
				if v.Symbol == 0 {
					continue
				}
				cs := NCS(v.Vector, q.Data)
				if cs > max {
					max, symbol = cs, v.Symbol
				}
			}
			query += fmt.Sprintf("%c", symbol)
			m.Add(symbol)
			i++
			if i >= 128 && (symbol == '.' || symbol == '!' || symbol == '?') {
				break
			}
			if i >= 1024 {
				break
			}
		}
		fmt.Printf(query)
		fmt.Println("\n++++++++++++++++++++++++++++++++++++++++")
	}
}
