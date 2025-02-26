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
)

// Query submits a query to the llm
func Query(query string) string {
	buffer := bytes.NewBuffer([]byte(fmt.Sprintf(`{ "model": "llama3.2", "prompt": "%s"}`, query)))
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

func main() {
	type Vector struct {
		Vector *[256]float32
		Symbol byte
	}
	mind, index := make([]Vector, 128*1024), 0
	m := NewMixer()
	m.Add(0)
	query := "Why is the sky blue?"
	for {
		answer := Query(query)
		if len(answer) != 0 {
			fmt.Printf(answer)
			fmt.Println("\n----------------------------------------")
			for _, v := range []byte(answer) {
				index = (index + 1) % len(mind)
				mind[index].Vector = m.Mix()
				mind[index].Symbol = v
				m.Add(v)
			}
		}
		query = ""
		for i := 0; i < 128; i++ {
			q := m.Mix()
			max, symbol := float32(0.0), byte(0)
			for _, v := range mind {
				if v.Symbol == 0 {
					continue
				}
				cs := CS(v.Vector[:], q[:])
				if cs > max {
					max, symbol = cs, v.Symbol
				}
			}
			query += fmt.Sprintf("%c", symbol)
			m.Add(symbol)
		}
		fmt.Printf(query)
		fmt.Println("\n++++++++++++++++++++++++++++++++++++++++")
	}
}
