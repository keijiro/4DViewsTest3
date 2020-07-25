using UnityEngine;
using Unity.Collections;

namespace Remesher {

//
// Voxelization - Triangles transform into boxes
//

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
public sealed class Voxelization : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region Public methods

    public void SampleSource()
    {
        OnDisable();
        OnDestroy();
    }

    #endregion

    #region Private objects

    NativeArray<VoxelizationEffect.Element> _voxels, _fragments;
    Mesh _mesh;

    #endregion

    #region MonoBehaviour implementation

    void OnDisable()
    {
        if (_voxels.IsCreated) _voxels.Dispose();
        if (_fragments.IsCreated) _fragments.Dispose();
    }

    void OnDestroy()
    {
        ObjectUtil.Destroy(_mesh);
        _mesh = null;
    }

    void LateUpdate()
    {
        // Lazy initialization
        if (!_voxels.IsCreated)
            (_voxels, _fragments) = VoxelizationEffect.
              Initialize(_source.sharedMesh, _source.transform);

        if (_mesh == null)
        {
            _mesh = MeshUtil.SetupWithMeshFilter(gameObject);
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10);
        }

        // Mesh reconstruction
        using (var vertices =
                 VoxelizationEffect.Build(_voxels, _fragments, _effector))
          MeshUtil.UpdateWithVertexArray(_mesh, vertices);
    }

    #endregion
}

}
