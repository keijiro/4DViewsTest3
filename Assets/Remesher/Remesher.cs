using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Mathematics;

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
sealed partial class Remesher : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;

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

        var sourceMesh = _source.sharedMesh;

        SetupMesh();

        using (var vertexArray = ArrayBuilder.CreateVertexArray(sourceMesh))
        using (var indexArray = ArrayBuilder.CreateIndexArray(sourceMesh))
          UpdateMesh(vertexArray, indexArray);

        _mesh.bounds = sourceMesh.bounds;
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
            (VertexAttribute.Position, VertexAttributeFormat.Float32, 3),
          new VertexAttributeDescriptor
            (VertexAttribute.Normal, VertexAttributeFormat.Float32, 3),
          new VertexAttributeDescriptor
            (VertexAttribute.Tangent, VertexAttributeFormat.Float32, 4),
          new VertexAttributeDescriptor
            (VertexAttribute.TexCoord0, VertexAttributeFormat.Float32, 2)
        );
        _mesh.SetVertexBufferData(vertexArray, 0, 0, vertexCount);

        _mesh.SetIndexBufferParams(vertexCount, IndexFormat.UInt32);
        _mesh.SetIndexBufferData(indexArray, 0, 0, vertexCount);

        _mesh.SetSubMesh(0, new SubMeshDescriptor(0, vertexCount));
    }

    #endregion
}
