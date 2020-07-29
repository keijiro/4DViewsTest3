using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.Timeline;
using Unity.Collections;
using Unity.Mathematics;
using Klak.Chromatics;

namespace Remesher {

//
// LightStrip - Self emissive line strip
//

[System.Serializable]
public struct LightStripConfig
{
    public int VertexCount;
    [Space]
    public float Radius;
    public float Height;
    public float2 Motion;
    [Space]
    public CosineGradient Gradient;
    public float GradientScroll;
    [Space]
    public float NoiseFrequency;
    public float NoiseMotion;
    public float NoiseAmplitude;
}

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
public sealed class LightStrip :
  MonoBehaviour, ITimeControl, IPropertyPreview
{
    #region Editable attributes

    [SerializeField] LightStripConfig _config = default(LightStripConfig);

    #endregion

    #region Private objects

    NativeArray<LightStripController.Element> _elements;
    Mesh _mesh;
    float _time;
    float _last;

    #endregion

    #region ITimeControl implementation

    public void OnControlTimeStart() => _time = 0;
    public void OnControlTimeStop() => _time = -1;
    public void SetTime(double time) => _time = (float)time;

    #endregion

    #region IPropertyPreview implementation

    public void GatherProperties
      (PlayableDirector director, IPropertyCollector driver) {}

    #endregion

    #region MonoBehaviour implementation

    void OnDisable()
    {
        if (_elements.IsCreated) _elements.Dispose();
    }

    void OnDestroy()
    {
        ObjectUtil.Destroy(_mesh);
        _mesh = null;
    }

    void LateUpdate()
    {
        if (_time < 0)
        {
            // Negative time: The module is to be disabled.
            OnDisable();
            OnDestroy();
            return;
        }

        // Lazy initialization
        if (!_elements.IsCreated)
        {
            _elements = LightStripController.Initialize(_config);
            _last = 0;
        }

        if (_mesh == null)
        {
            _mesh = MeshUtil.SetupWithMeshFilter(gameObject);
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10);
        }

        // Time update
        // (We don't support rewinding at the moment.)
        if (_time > _last)
        {
            var dt = (_time - _last) / 3;
            LightStripController.Update(_config, _elements, _time, dt);
            LightStripController.Update(_config, _elements, _time + dt, dt);
            LightStripController.Update(_config, _elements, _time + dt * 2, dt);
        }

        _last = _time;

        // Mesh reconstruction
        using (var vertices = LightStripController.BuildVertexArray(_elements))
          using (var indices = LightStripController.BuildIndexArray(_elements))
            MeshUtil.UpdateWithVertexIndexArrays(_mesh, vertices, indices);
    }

    #endregion
}

}