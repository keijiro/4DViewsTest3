using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.Timeline;
using Unity.Collections;

namespace Remesher {

//
// Disintegration - Flat shader + per-triangle disintegration effect
//

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
public sealed class Disintegration :
  MonoBehaviour, ITimeControl, IPropertyPreview
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region Private objects

    NativeArray<DisintegrationEffect.Fragment> _fragments;
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
        if (_fragments.IsCreated) _fragments.Dispose();
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
        if (!_fragments.IsCreated)
        {
            _fragments = DisintegrationEffect.Initialize
              (_source.sharedMesh, _source.transform);
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
            DisintegrationEffect.Update(_fragments, _effector, _time - _last);
        _last = _time;

        // Mesh reconstruction
        using (var vertices = DisintegrationEffect.Build(_fragments))
          MeshUtil.UpdateWithVertexArray(_mesh, vertices);
    }

    #endregion
}

}
