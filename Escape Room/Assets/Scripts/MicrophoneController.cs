using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MelGram;
using TensorFlow;
using UnityEngine;
using UnityEngine.Assertions.Comparers;
using UnityEngine.Experimental.Audio;
using System.Threading;
using Random = System.Random;

public class MicrophoneController : MonoBehaviour {
    public float Sensitivity = 100;
    private float _loudness;
    private AudioSource _audioSource;
    private string _microphone;

    public TextAsset GraphModel;

    private float _recognitionInterval = 0.2f;
    private float _recognitionCounter = 0;


    private TFGraph _graph;
    private TFSession _session;

    private int _frameLenght = 4410;

    private int _fs = 44100;
    private int _bufferLenght = 44100;
    private float[] _ringBuffer;
    private List<float> _bufferOverflow;
    private int _previousSampleLenght = 0;

    private Queue<float> _audioBuffer;

    private bool _threadRunning;
    private Dictionary<ulong, Thread> _threads;
    private ulong _threadCounter = 0;


    // Start is called before the first frame update
    void Start() {
        _audioSource = GetComponent<AudioSource>();
        _ringBuffer = new float[_bufferLenght];
        _bufferOverflow = new List<float>();

        _audioBuffer = new Queue<float>();

        // We get the first available microphone
        foreach (var device in Microphone.devices) {
            Debug.Log(device);
            if (_microphone == null) {
                _microphone = device;
            }
        }

        // TODO: Add error if no microphone detected?

        _audioSource.clip = Microphone.Start(_microphone, true, 10, _fs);
        _audioSource.loop = true;

        if (Microphone.IsRecording(_microphone)) {
            // Wait until the recording has started
            while (!(Microphone.GetPosition(_microphone) > 0)) { }
            _audioSource.Play();
        }

        // Init graph
        
        // Warning: Heavy to load! Consider loading screen?
        _graph = new TFGraph();
        _graph.Import(GraphModel.bytes);
        _session = new TFSession(_graph);
        _threads = new Dictionary<ulong, Thread>();

    }

    // Update is called once per frame
    void Update() {
        //if (_recognitionCounter < _recognitionInterval) {
        //    _recognitionCounter += Time.deltaTime;
        //} else {
        //    _recognitionCounter = 0;
        //}

        // Ring buffer
        var sampleLenght = (int)(_fs * Time.deltaTime);
        var data = new float[sampleLenght];
        _audioSource.GetOutputData(data, 0);
        for (int i = sampleLenght - 1; i > -1; i--) {
            _audioBuffer.Enqueue(data[i]);
        }

        if (_audioBuffer.Count > _bufferLenght) {
            Predict();
            Debug.Log("Current threads " + _threads.Count);
        }
    }

    void Predict() {
        var tensorData = new float[1, _bufferLenght, 1];
        float maxData = -1;
        for (int i = 0; i < _ringBuffer.Length; i++) {
            float data = _audioBuffer.Dequeue();
            tensorData[0, i, 0] = data;
            if (maxData < data) {
                maxData = data;
            }
        }

        
        if (maxData > 0.2)
        {
            Thread newThread = new Thread(() => PredictAux(_threadCounter, tensorData));
            _threads.Add(_threadCounter, newThread);
            _threadCounter++;
            newThread.Start();
        }
    }

    void PredictAux(ulong id, float[,,] tensorData) 
    {
        var runner = _session.GetRunner();
        TFTensor input = tensorData;
        runner.AddInput(_graph["input_node_input"][0], input);
        runner.Fetch(_graph["output_node/BiasAdd"][0]);

        var recurrentTensor = runner.Run()[0].GetValue() as float[,];

        // We dispose of resources so our graph doesn't break down over
        // time. IMPORTANT if you will repeatedly call the graph.
        //session.Dispose();
        //graph.Dispose();

        //int c = 0;
        //foreach (var v in recurrentTensor) {
        //    Debug.Log("output: " + c++);
        //    Debug.Log(v);
        //}

        int c = 0;
        List<float> outputs = new List<float>();
        foreach (var v in recurrentTensor) {
            outputs.Add(v);
        }
        string e = "";
        if (outputs[0] > outputs[1] && outputs[0] > 0.1) {
            e = "clap";
        } else if (outputs[1] > outputs[0] && outputs[1] > 0.1) {
            e = "keys";
        }
        Debug.Log("Estimated: " + e + " | " + outputs[0] + " " + outputs[1]);
        _threads.Remove(id);
    }

    void OnDisable() {
        // Wait for each thread to terminate
        foreach (var thread in _threads)
        {
            thread.Value.Abort();
        }
    }
}
