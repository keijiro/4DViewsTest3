float4 _Color1, _Color2, _Color3, _Color4;
float _Param1, _Param2, _Param3, _Param4;
float _LocalTime;

float3 WallFx(float2 uv);

void Vertex
  (float4 vertex : POSITION,
   float2 texcoord : TEXCOORD0,
   out float4 out_vertex : SV_Position,
   out float2 out_texcoord : TEXCOORD0)
{
    out_vertex = float4(vertex.x * 2 - 1, 1 - vertex.y * 2, 1, 1);
    out_texcoord = texcoord;
}

float4 Fragment
  (float4 vertex : SV_Position,
   float2 texcoord : TEXCOORD0) : SV_Target
{
    return float4(WallFx(texcoord), 1);
}
