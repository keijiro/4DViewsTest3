using Unity.Mathematics;

namespace Remesher {

struct Vertex
{
    public float3 Position;
    public float3 Normal;
    public float4 Tangent;
    public float4 Color;
    public float4 TexCoord;

    public const int StructSize = sizeof(float) * 18;

    public Vertex(float3 position, float3 normal, float4 tangent,
                  float4 color, float4 texCoord)
    {
        Position = position;
        Normal = normal;
        Tangent = tangent;
        Color = color;
        TexCoord = texCoord;
    }
}

struct Triangle
{
    public Vertex Vertex1;
    public Vertex Vertex2;
    public Vertex Vertex3;

    public const int StructSize = Vertex.StructSize * 3;

    public Triangle(in Vertex v1, in Vertex v2, in Vertex v3)
    {
        Vertex1 = v1;
        Vertex2 = v2;
        Vertex3 = v3;
    }
}

}
