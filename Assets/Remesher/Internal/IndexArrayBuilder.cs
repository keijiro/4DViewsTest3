using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

namespace Remesher { 

static class IndexArrayBuilder
{
    public static NativeArray<uint> SimpleArray(int count)
    {
        var array = MemoryUtil.TempJobArray<uint>(count);
        new SimpleArrayJob { Output = array, Count = count }.Run();
        return array;
    }

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct SimpleArrayJob : IJob
    {
        [WriteOnly] public NativeArray<uint> Output;

        public int Count;

        public void Execute()
        {
            for (var i = 0; i < Count; i++) Output[i] = (uint)i;
        }
    }
}

}
