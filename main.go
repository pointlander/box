// Copyright 2025 The Box Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"

	"github.com/pointlander/gradient/tf32"

	"github.com/alixaxel/pagerank"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
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
		s, is, c := make(map[byte]int), make(map[int]byte), 0
		for _, v := range []byte(answer) {
			if _, has := s[v]; !has {
				s[v] = c
				is[c] = v
				c++
			}
		}
		others := tf32.NewSet()
		others.Add("input", 8*256, len([]byte(answer)))
		others.Add("output", len(s), len([]byte(answer)))
		if len(answer) != 0 {
			input, output := others.ByName["input"], others.ByName["output"]
			fmt.Printf(answer)
			fmt.Println("\n----------------------------------------")
			for _, v := range []byte(answer) {
				index = (index + 1) % len(mind)
				vectors := m.Mix()
				mind[index].Vector = vectors.Data
				mind[index].Symbol = v
				input.X = append(input.X, vectors.Data...)
				out := make([]float32, len(s))
				out[s[v]] = 1
				output.X = append(output.X, out...)
				m.Add(v)
			}
		}
		rng := rand.New(rand.NewSource(1))
		set := tf32.NewSet()
		set.Add("w", 8*256, len(s))
		set.Add("b", len(s))
		set.Add("w1", len(s), len(s))
		set.Add("b1", len(s))
		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float32, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float32, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rng.NormFloat64()*factor))
			}
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
		}
		l := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w"), others.Get("input")), set.Get("b")))
		l1 := tf32.Add(tf32.Mul(set.Get("w1"), l), set.Get("b1"))
		loss := tf32.Avg(tf32.Quadratic(l1, others.Get("output")))
		for i := 0; i < 256; i++ {
			pow := func(x float32) float32 {
				y := math.Pow(float64(x), float64(i+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return float32(y)
			}

			others.Zero()
			set.Zero()

			cost := tf32.Gradient(loss).X[0]
			if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
				break
			}

			norm := float32(0.0)
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			b1, b2 := pow(B1), pow(B2)
			scaling := float32(1.0)
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set.Weights {
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
				}
			}
			fmt.Println(i, cost)
		}

		fmt.Println("done training")
		others = tf32.NewSet()
		others.Add("input", 8*256, 1)
		input := others.ByName["input"]
		input.X = input.X[:cap(input.X)]
		l = tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w"), others.Get("input")), set.Get("b")))
		l1 = tf32.Add(tf32.Mul(set.Get("w1"), l), set.Get("b1"))
		type Path struct {
			Path string
			Cost float32
		}
		paths := make([]Path, 0, 8)
		for j := 0; j < 128; j++ {
			query := ""
			cost := float32(0.0)
			i := 0
			cp := m.Copy()
			for {
				q := cp.Mix()
				for i, v := range q.Data {
					input.X[i] = v
				}
				max, symbol := float32(0.0), byte(0)
				l1(func(a *tf32.V) bool {
					sum := float32(0.0)
					for _, v := range a.X {
						sum += v
					}
					selection, total := rng.Float32(), float32(0.0)
					for i, v := range a.X {
						total += v / sum
						if selection < total {
							cost += v / sum
							symbol = is[i]
							break
						}
						/*if v > max {
							max, symbol = v, is[i]
						}*/
					}
					return true
				})
				_ = max
				query += fmt.Sprintf("%c", symbol)
				cp.Add(symbol)
				i++
				if i >= 128 && (symbol == '.' || symbol == '!' || symbol == '?') {
					break
				}
				if i >= 1024 {
					break
				}
			}
			paths = append(paths, Path{
				Path: query,
				Cost: cost,
			})
		}
		for i := range paths {
			paths[i].Cost /= float32(len(paths[i].Path))
		}
		sort.Slice(paths, func(i, j int) bool {
			return paths[i].Cost > paths[j].Cost
		})
		fmt.Printf(paths[0].Path)
		fmt.Println("\n****************************************")

		query = ""
		i := 0
		for {
			q := m.Mix()
			max, symbol := float32(0.0), byte(0)
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
