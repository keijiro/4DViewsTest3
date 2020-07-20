using UnityEngine;
using Unity.Collections;

namespace Remesher {

//
// Triangulation - Flat shade + simple vertex effects
//

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
sealed partial class Triangulation : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region MonoBehaviour implementation

    Mesh _mesh;

    void OnDestroy()
    {
        ObjectUtil.Destroy(_mesh);
        _mesh = null;
    }

    void LateUpdate()
    {
        if (_source == null || _source.sharedMesh == null) return;
        if (_effector == null) return;

        if (_mesh == null) _mesh = MeshUtil.SetupWithMeshFilter(gameObject);

        var args = new ArrayBuilder.Arguments
          (_source.sharedMesh, _source.transform, _effector);

        using (var vertices = ArrayBuilder.CreateVertexArray(args))
        using (var indices = ArrayBuilder.CreateIndexArray(args))
          MeshUtil.UpdateWithArrays(_mesh, vertices, indices);

        _mesh.bounds = GeomUtil.TransformBounds
          (_source.sharedMesh.bounds, _source.transform);
    }

    #endregion
}

}
