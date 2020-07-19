using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;

//
// Remesher is a MonoBehaviour class for adding vertex modification effecs
// using C# job system and the new Mesh accessing API. This is the only
// effective way to modify vertices in the DXR pipeline because it's very
// troublesome to write a custom vertex/geometry shader that works properly
// with DXR.
//

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
sealed partial class Remesher : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region MonoBehaviour implementation

    void OnDestroy()
    {
        if (_mesh != null)
        {
            if (Application.isPlaying)
                Destroy(_mesh);
            else
                DestroyImmediate(_mesh);
        }
        _mesh = null;
    }

    void LateUpdate()
    {
        if (_source == null || _source.sharedMesh == null) return;
        if (_effector == null) return;

        SetupMesh();

        var args = new Arguments
          (_source.sharedMesh, _source.transform, _effector);

        using (var vertexArray = ArrayBuilder.CreateVertexArray(args))
        using (var indexArray = ArrayBuilder.CreateIndexArray(args))
          UpdateMesh(vertexArray, indexArray);

        _mesh.bounds = CalculateTransformedBounds
          (_source.sharedMesh.bounds, _source.transform);
    }

    #endregion

    #region Internal-use method

    static Bounds CalculateTransformedBounds
      (Bounds bounds, Transform transform)
    {
        var center = transform.TransformPoint(bounds.center);
        var size = transform.TransformVector(bounds.size);
        var maxs = Mathf.Max(Mathf.Max(size.x, size.y), size.z);
        return new Bounds(center, Vector3.one * maxs);
    }

    #endregion

    #region Mesh object operations

    Mesh _mesh;

    void SetupMesh()
    {
        if (_mesh != null) return;

        _mesh = new Mesh();
        _mesh.hideFlags = HideFlags.DontSave;
        _mesh.MarkDynamic();

        var mf = GetComponent<MeshFilter>();
        if (mf == null)
        {
            mf = gameObject.AddComponent<MeshFilter>();
            mf.hideFlags = HideFlags.NotEditable | HideFlags.DontSave;
        }

        mf.sharedMesh = _mesh;
    }

    void UpdateMesh
      (NativeArray<Vertex> vertexArray, NativeArray<uint> indexArray)
    {
        _mesh.Clear();

        var vertexCount = vertexArray.Length;

        _mesh.SetVertexBufferParams(
          vertexCount,
          new VertexAttributeDescriptor
            (VertexAttribute.Position  , VertexAttributeFormat.Float32, 3),
          new VertexAttributeDescriptor
            (VertexAttribute.Normal    , VertexAttributeFormat.Float32, 3),
          new VertexAttributeDescriptor
            (VertexAttribute.Tangent   , VertexAttributeFormat.Float32, 4),
          new VertexAttributeDescriptor
            (VertexAttribute.TexCoord0 , VertexAttributeFormat.Float32, 2)
        );
        _mesh.SetVertexBufferData(vertexArray, 0, 0, vertexCount);

        _mesh.SetIndexBufferParams(vertexCount, IndexFormat.UInt32);
        _mesh.SetIndexBufferData(indexArray, 0, 0, vertexCount);

        _mesh.SetSubMesh(0, new SubMeshDescriptor(0, vertexCount));
    }

    #endregion
}
