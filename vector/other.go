// Copyright 2025 The Box Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386
// +build 386

package vector

func Dot(x, y []float32) float32 {
	return dot(x, y)
}
