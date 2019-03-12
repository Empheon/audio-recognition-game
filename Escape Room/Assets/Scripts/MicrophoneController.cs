using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions.Comparers;
using UnityEngine.Experimental.Audio;

public class MicrophoneController : MonoBehaviour
{
    public float Sensitivity = 100;
    private float _loudness;
    private AudioSource _audioSource;
    private string _microphone;

    // Start is called before the first frame update
    void Start()
    {
        _audioSource = GetComponent<AudioSource>();

        // We get the first available microphone
        foreach (var device in Microphone.devices)
        {
            if (_microphone == null)
            {
                _microphone = device;
            }
        }

        // TODO: Add error if no microphone detected?

        _audioSource.clip = Microphone.Start(_microphone, true, 10, 44100);
        _audioSource.loop = true;

        if (Microphone.IsRecording(_microphone))
        {
            // Wait until the recording has started
            while (!(Microphone.GetPosition(_microphone) > 0)) { }
            _audioSource.Play();
        }
    }

    // Update is called once per frame
    void Update()
    {
        // TODO: Feed data to the graph
        //_loudness = GetAveragedVolume() * Sensitivity;
        //Debug.Log(_loudness);
    }

    float GetAveragedVolume()
    {
        var data = new float[256];
        _audioSource.GetOutputData(data, 0);
        var sum = 0f;
        foreach (var sample in data) {
            sum += sample;
        }
        return sum / 256;
    }
}
