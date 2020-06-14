// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.benchmarkncnn;

import android.content.res.AssetManager;

public class BenchmarkNcnn
{
    public native boolean Init();

    public native String GetPlatform();

    public class Obj
    {
        // 0 = success
        // 1 = no gpu
        public int retcode;
        public float min;
        public float max;
        public float avg;
    }

    public native Obj Run(AssetManager mgr, int threads, int powersave,
                          boolean mempool, boolean winograd, boolean sgemm, boolean pack4, boolean bf16s,
                          boolean gpu, boolean gpufp16p, boolean gpufp16s, boolean gpufp16a, boolean gpupack8,
                          int model, int loops);

    static {
        System.loadLibrary("benchmarkncnn");
    }
}
