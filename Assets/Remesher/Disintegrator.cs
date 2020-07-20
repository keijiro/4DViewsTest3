using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;

namespace Remesher {

[ExecuteInEditMode, RequireComponent(typeof(MeshRenderer))]
sealed partial class Disintegrator : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] MeshFilter _source = null;
    [SerializeField] Transform _effector = null;

    #endregion

    #region Public method

    public void Kick()
    {
        OnDestroy();
        InitializeFragments();
    }

    #endregion

    #region MonoBehaviour implementation

    void OnDisable()
    {
        if (_fragments.IsCreated) _fragments.Dispose();
    }

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
        if (!_fragments.IsCreated) return;

        if (_mesh == null) SetupMesh();

        using (var vertexArray = FragmentController.CreateVertexArray(_fragments))
        using (var indexArray = FragmentController.CreateIndexArray(_fragments))
          UpdateMesh(vertexArray, indexArray);
    }

    #endregion

    #region Fragment array

    NativeArray<Fragment> _fragments;

    void InitializeFragments()
    {
        _fragments = FragmentController.Initialize
          (_source.sharedMesh, _source.transform);
    }

    #endregion

    #region Mesh object operations

    Mesh _mesh;

    void SetupMesh()
    {
        if (_mesh != null) return;

        _mesh = new Mesh();
        _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10);
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

}
