void DetectEdge_float(float3 coords, float width, out float edge)
{
    float3 fw = fwidth(coords) * width;
    float3 v = saturate(1 - coords / fw);
    edge = max(max(v.x, v.y), v.z);
}
