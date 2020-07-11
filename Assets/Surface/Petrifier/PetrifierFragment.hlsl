// Petrifier effect fragment shader
// https://github.com/keijiro/TestbedHDRP

float _TriplanarTiling;
float4 _TriplanarBaseColor;
Texture2D _TriplanarColorMap;
Texture2D _TriplanarNormalMap;
SamplerState sampler_TriplanarColorMap;
SamplerState sampler_TriplanarNormalMap;

void CustomTriplanar(float3 wpos, float3 wnrm, out float4 color, out float3 normal)
{
    wpos = GetAbsolutePositionWS(wpos) * _TriplanarTiling;

    float2 tx = wpos.yz;
    float2 ty = wpos.zx;
    float2 tz = wpos.xy;

    float3 bf = normalize(abs(wnrm));
    bf /= dot(bf, 1);

    // Base color
    half4 cx = SAMPLE_TEXTURE2D(_TriplanarColorMap, sampler_TriplanarColorMap, tx) * bf.x;
    half4 cy = SAMPLE_TEXTURE2D(_TriplanarColorMap, sampler_TriplanarColorMap, ty) * bf.y;
    half4 cz = SAMPLE_TEXTURE2D(_TriplanarColorMap, sampler_TriplanarColorMap, tz) * bf.z;
    color = (cx + cy + cz) * _TriplanarBaseColor;

    // Normal map
	float3 nx = UnpackNormalmapRGorAG(SAMPLE_TEXTURE2D(_TriplanarNormalMap, sampler_TriplanarNormalMap, tx)) * bf.x;
	float3 ny = UnpackNormalmapRGorAG(SAMPLE_TEXTURE2D(_TriplanarNormalMap, sampler_TriplanarNormalMap, ty)) * bf.y;
	float3 nz = UnpackNormalmapRGorAG(SAMPLE_TEXTURE2D(_TriplanarNormalMap, sampler_TriplanarNormalMap, tz)) * bf.z;
    normal = normalize(nx + ny + nz);
}

// Fragment shader function, copy-pasted from HDRP/ShaderPass/ShaderPassGBuffer.hlsl
// There are a few modification from the original shader. See "Custom:" for details.
void PetrifierFragment(
            PackedVaryingsToPS packedInput,
            OUTPUT_GBUFFER(outGBuffer)
            #ifdef _DEPTHOFFSET_ON
            , out float outputDepth : SV_Depth
            #endif
            )
{
    FragInputs input = UnpackVaryingsMeshToFragInputs(packedInput.vmesh);

    // input.positionSS is SV_Position
    PositionInputs posInput = GetPositionInput(input.positionSS.xy, _ScreenSize.zw, input.positionSS.z, input.positionSS.w, input.positionRWS);

#ifdef VARYINGS_NEED_POSITION_WS
    float3 V = GetWorldSpaceNormalizeViewDir(input.positionRWS);
#else
    // Unused
    float3 V = float3(1.0, 1.0, 1.0); // Avoid the division by 0
#endif

    SurfaceData surfaceData;
    BuiltinData builtinData;
    GetSurfaceAndBuiltinData(input, V, posInput, surfaceData, builtinData);

    // Custom: Triplanar mapping
    float4 color;
    float3 normal;
    CustomTriplanar(posInput.positionWS, surfaceData.normalWS, color, normal);
    surfaceData.baseColor = Luminance(surfaceData.baseColor);
    surfaceData.baseColor = lerp(surfaceData.baseColor, color.rgb, color.a);
	surfaceData.normalWS = normalize(TransformTangentToWorld(normal, input.tangentToWorld));

    ENCODE_INTO_GBUFFER(surfaceData, builtinData, posInput.positionSS, outGBuffer);

#ifdef _DEPTHOFFSET_ON
    outputDepth = posInput.deviceDepth;
#endif
}
