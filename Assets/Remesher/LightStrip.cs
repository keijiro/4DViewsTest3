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
    public float Radius;
    public float Height;
    public float Thickness;
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
    [Space]
    [SerializeField] int _vertexCount = 100;
    [SerializeField] float _timeStep = 1.0f / 60;

    #endregion

    #region Private objects

    NativeArray<LightStripController.Element> _elements;
    Mesh _mesh;
    float _time;
    float _last;

    #endregion

    #region ITimeControl implementation

    public void OnControlTimeStart() {}
    public void OnControlTimeStop() {}
    public void SetTime(double time) => _time = (float)time;

    #endregion

    #region IPropertyPreview implementation

    public void GatherProperties
      (PlayableDirector director, IPropertyCollector driver) {}

    #endregion

    #region MonoBehaviour implementation

    void OnValidate()
    {
        _vertexCount = math.max(_vertexCount, 8);
        _timeStep = math.max(_timeStep, 1.0f / 60 / 10);

        // Make the mesh reset on the next update.
        _last = 1e+10f;
    }

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
        // Dispose the current state if _time is invalid.
        if (_time < _last)
        {
            OnDisable();
            OnDestroy();
        }

        // Lazy initialization
        if (!_elements.IsCreated)
        {
            _elements = LightStripController.
              Initialize(_config, _vertexCount, _time, _timeStep);
            _last = _time;
        }

        if (_mesh == null)
        {
            _mesh = MeshUtil.SetupWithMeshFilter(gameObject);
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10);
        }

        // Time advance steps
        var dt = math.max((_time - _last) / 10, _timeStep);

        while (_time - _last > dt)
        {
            LightStripController.Update(_config, _elements, _last, dt);
            _last += dt;
        }

        // Last step
        if (_time > _last)
        {
            LightStripController.
              Update(_config, _elements, _last, _time - _last);
            _last = _time;
        }

        // Mesh reconstruction
        using (var vertices =
               LightStripController.BuildVertexArray(_config, _elements))
          using (var indices =
                 LightStripController.BuildIndexArray(_config, _elements))
            MeshUtil.UpdateWithVertexIndexArrays(_mesh, vertices, indices);
    }

    #endregion
}

}
