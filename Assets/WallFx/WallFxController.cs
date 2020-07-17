using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.Timeline;

[ExecuteInEditMode]
sealed class WallFxController : MonoBehaviour, ITimeControl, IPropertyPreview
{
    #region Editable attributes

    [SerializeField] Shader _shader = null;
    [SerializeField] RenderTexture _target = null;

    [SerializeField] Color _color1 = Color.red;
    [SerializeField] Color _color2 = Color.blue;
    [SerializeField] Color _color3 = Color.green;
    [SerializeField] Color _color4 = Color.white;

    [SerializeField] float _param1 = 0;
    [SerializeField] float _param2 = 0;
    [SerializeField] float _param3 = 0;
    [SerializeField] float _param4 = 0;

    #endregion

    #region ITimeControl implementation

    float _time;

    public void OnControlTimeStart() {}
    public void OnControlTimeStop() {}
    public void SetTime(double time) => _time = (float)time;

    #endregion

    #region IPropertyPreview implementation

    public void GatherProperties
      (PlayableDirector director, IPropertyCollector driver) {}

    #endregion

    #region MonoBehaviour implementation

    Material _material;

    void OnDestroy()
    {
        if (_material != null)
        {
            if (Application.isPlaying)
                Destroy(_material);
            else
                DestroyImmediate(_material);
        }
        _material = null;
    }

    void LateUpdate()
    {
        if (_shader == null && _target == null) return;

        if (_material == null)
        {
            _material = new Material(_shader);
            _material.hideFlags = HideFlags.DontSave;
        }

        _material.SetColor("_Color1", _color1);
        _material.SetColor("_Color2", _color2);
        _material.SetColor("_Color3", _color3);
        _material.SetColor("_Color4", _color4);

        _material.SetFloat("_Param1", _param1);
        _material.SetFloat("_Param2", _param2);
        _material.SetFloat("_Param3", _param3);
        _material.SetFloat("_Param4", _param4);

        _material.SetFloat("_LocalTime", _time);

        Graphics.Blit(null, _target, _material, 0);
    }

    #endregion
}
