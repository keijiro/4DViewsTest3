// Petrifier effect geometry shader
// https://github.com/keijiro/TestbedHDRP

// Vertex output from geometry
PackedVaryingsType VertexOutput
  (AttributesMesh source, float3 position, half3 normal)
{
    return PackVertexData(source, position, normal, 1);
}

// Geometry shader function body
[maxvertexcount(3)]
void PetrifierGeometry(
  uint pid : SV_PrimitiveID,
  triangle Attributes input[3],
  inout TriangleStream<PackedVaryingsType> outStream
)
{
    // Input vertices
    AttributesMesh v0 = ConvertToAttributesMesh(input[0]);
    AttributesMesh v1 = ConvertToAttributesMesh(input[1]);
    AttributesMesh v2 = ConvertToAttributesMesh(input[2]);

    float3 p0 = v0.positionOS;
    float3 p1 = v1.positionOS;
    float3 p2 = v2.positionOS;

#ifdef ATTRIBUTES_NEED_NORMAL
    float3 n0 = v0.normalOS;
    float3 n1 = v1.normalOS;
    float3 n2 = v2.normalOS;
#else
    float3 n0 = 0;
    float3 n1 = 0;
    float3 n2 = 0;
#endif

    n0 = n1 = n2 = normalize(cross(p1 - p0, p2 - p0));

    outStream.Append(VertexOutput(v0, p0, n0));
    outStream.Append(VertexOutput(v1, p1, n1));
    outStream.Append(VertexOutput(v2, p2, n2));
    outStream.RestartStrip();
}
