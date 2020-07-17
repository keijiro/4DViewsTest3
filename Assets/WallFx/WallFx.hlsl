#include "Packages/jp.keijiro.noiseshader/Shader/SimplexNoise2D.hlsl"

float3 WallFx(float2 uv)
{
    const float freq = _Param1;
    const float speed = _Param2;
    const float width = _Param3;

    float x = uv.x * freq;
    float t1 = _LocalTime * speed;
    float t2 = -10 - t1;

    float2 np1 = float2(x, t1);
    float2 np2 = float2(x, t2);

    float n1 = abs(snoise(np1)) < width;
    float n2 = abs(snoise(np2)) < width;

    return _Color1.rgb * n1 + _Color2.rgb * n2;
}
