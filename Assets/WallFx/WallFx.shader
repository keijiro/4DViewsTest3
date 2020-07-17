Shader "Hidden/WallFx"
{
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            HLSLPROGRAM
            #pragma vertex Vertex
            #pragma fragment Fragment
            #include "Common.hlsl"
            #include "WallFx.hlsl"
            ENDHLSL
        }
    }
}
