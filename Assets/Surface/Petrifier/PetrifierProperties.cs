using UnityEngine;

[ExecuteInEditMode]
class PetrifierProperties : MonoBehaviour
{
    [SerializeField] Color _baseColor = Color.white;
    [SerializeField] Texture2D _colorMap = null;
    [SerializeField] Texture2D _normalMap = null;
    [SerializeField] float _tiling = 1;

    MaterialPropertyBlock _sheet;
    Renderer _renderer;

    void LateUpdate()
    {
        if (_sheet == null) _sheet = new MaterialPropertyBlock();
        if (_renderer == null) _renderer = GetComponent<Renderer>();

        _renderer.GetPropertyBlock(_sheet);

        _sheet.SetColor("_TriplanarBaseColor", _baseColor);

        if (_colorMap != null)
            _sheet.SetTexture("_TriplanarColorMap", _colorMap);

        if (_normalMap != null)
            _sheet.SetTexture("_TriplanarNormalMap", _normalMap);

        _sheet.SetFloat("_TriplanarTiling", _tiling);

        _renderer.SetPropertyBlock(_sheet);
    }
}
