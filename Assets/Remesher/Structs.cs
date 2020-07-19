using Unity.Mathematics;

namespace Remesher {

struct Vertex
{
    public float3 Position;
    public float3 Normal;
    public float4 Tangent;
    public float2 TexCoord;

    public Vertex(float3 position, float3 normal,
                  float4 tangent, float2 texCoord)
    {
        Position = position;
        Normal = normal;
        Tangent = tangent;
        TexCoord = texCoord;
    }
}

struct Triangle
{
    public Vertex Vertex1;
    public Vertex Vertex2;
    public Vertex Vertex3;

    public Triangle(in Vertex v1, in Vertex v2, in Vertex v3)
    {
        Vertex1 = v1;
        Vertex2 = v2;
        Vertex3 = v3;
    }
}

}
