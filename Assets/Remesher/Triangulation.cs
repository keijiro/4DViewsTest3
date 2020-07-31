using UnityEngine;
using Unity.Mathematics;

namespace Remesher {

//
// Triangulation - Flat shade + simple vertex effects
//

[System.Serializable]
public struct TriangulationConfig
{
    public int EffectType;
    public float3 ScaleParams;
    [Range(0, 1)] public float Softness;
    [Range(0, 1)] public float Probability;
}

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
public sealed class Triangulation : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] TriangulationConfig _config = default(TriangulationConfig);
    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region Private objects

    Mesh _mesh;

    #endregion

    #region MonoBehaviour implementation

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

        using (var vertices = TriangulationEffect.Build
                (_config, _source.sharedMesh, _source.transform, _effector))
          MeshUtil.UpdateWithVertexArray(_mesh, vertices);

        _mesh.bounds = GeomUtil.TransformBounds
          (_source.sharedMesh.bounds, _source.transform);
    }

    #endregion
}

}
