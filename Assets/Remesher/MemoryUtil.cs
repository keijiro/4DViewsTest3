using Unity.Collections;

namespace Remesher {

static class MemoryUtil
{
    public static NativeArray<T> Array<T>(int length) where T : struct
        => new NativeArray<T>(length, Allocator.Persistent,
                              NativeArrayOptions.UninitializedMemory);

    public static NativeArray<T> TempArray<T>(int length) where T : struct
        => new NativeArray<T>(length, Allocator.Temp,
                              NativeArrayOptions.UninitializedMemory);

    public static NativeArray<T> TempJobArray<T>(int length) where T : struct
        => new NativeArray<T>(length, Allocator.TempJob,
                              NativeArrayOptions.UninitializedMemory);
}

}
