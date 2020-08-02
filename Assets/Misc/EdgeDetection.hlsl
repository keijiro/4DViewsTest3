#if SHADERPASS == SHADERPASS_RAYTRACING_INDIRECT || \
    SHADERPASS == SHADERPASS_RAYTRACING_FORWARD || \
    SHADERPASS == SHADERPASS_RAYTRACING_GBUFFER || \
    SHADERPASS == SHADERPASS_RAYTRACING_VISIBILITY || \
    SHADERPASS == SHADERPASS_RAYTRACING_SUB_SURFACE
#define DETECTEDGE_FWIDTH(x) 0.01
#else
#define DETECTEDGE_FWIDTH(x) fwidth(x)
#endif

void DetectEdge_float(float3 coords, float width, out float edge)
{
    float3 fw = DETECTEDGE_FWIDTH(coords) * width;
    float3 v = saturate(1 - coords / fw);
    edge = max(max(v.x, v.y), v.z);
}

