using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;

public class NoteManager : MonoBehaviour
{
    public PostProcessProfile PostProcessProfile;
    private DepthOfField _depthOfField;
    // Start is called before the first frame update
    void Awake()
    {
        _depthOfField = PostProcessProfile.GetSetting<DepthOfField>();
    }

    void OnEnable()
    {
        _depthOfField.active = true;
    }

    void OnDisable()
    {
        _depthOfField.active = false;
    }
}
