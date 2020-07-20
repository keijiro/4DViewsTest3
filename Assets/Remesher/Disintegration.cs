using UnityEngine;
using Unity.Collections;

namespace Remesher {

//
// Disintegration - Flat shader + per-triangle disintegration effect
//

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
public sealed class Disintegration : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region Public method

    public void Kick()
    {
        if (_fragments.IsCreated) _fragments.Dispose();

        _fragments = DisintegrationEffect.Initialize
          (_source.sharedMesh, _source.transform);
    }

    #endregion

    #region Private objects

    NativeArray<DisintegrationEffect.Fragment> _fragments;
    Mesh _mesh;

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
        if (!_fragments.IsCreated) return;

        if (_mesh == null)
        {
            _mesh = MeshUtil.SetupWithMeshFilter(gameObject);
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10);
        }

        using (var vertices = DisintegrationEffect.Build(_fragments))
          MeshUtil.UpdateWithVertexArray(_mesh, vertices);
    }

    #endregion
}

}
