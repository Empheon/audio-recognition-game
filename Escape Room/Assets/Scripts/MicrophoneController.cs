using System.Collections.Generic;
using System.Linq;
using TensorFlow;
using UnityEngine;
using System.Threading;
using Accord.Audio;
using Accord.Math;

public class MicrophoneController : MonoBehaviour {
    private AudioSource _audioSource;
    private string _microphone;

    public TextAsset GraphModel;

    private TFGraph _graph;
    private TFSession _session;

    private const int Fs = 16000;
    private const int BufferLenght = 16000;
    private const int FrameRate = 50;
    private const int CeptrumCount = 13;
    private Queue<float> _audioBuffer;
    private MelFrequencyCepstrumCoefficient _mfcc;

    //private Dictionary<ulong, Thread> _threads;

    private readonly AudioAction[] _audioActions = {AudioAction.CLAP, AudioAction.KEYS, AudioAction.BAG};
    private ActionController _actionController;

    // Start is called before the first frame update
    void Start() {
        _audioSource = GetComponent<AudioSource>();
        _audioBuffer = new Queue<float>();
        _actionController = GetComponent<ActionController>();

        // We get the first available microphone
        foreach (var device in Microphone.devices) {
            Debug.Log(device);
            if (_microphone == null) {
                _microphone = device;
            }
        }

        // TODO: Add error if no microphone detected?

        _audioSource.clip = Microphone.Start(_microphone, true, 10, Fs);
        _audioSource.loop = true;

        if (Microphone.IsRecording(_microphone)) {
            // Wait until the recording has started
            while (!(Microphone.GetPosition(_microphone) > 0)) { }
            _audioSource.Play();
        }

        // Init graph
        
        _graph = new TFGraph();
        _graph.Import(GraphModel.bytes);
        _session = new TFSession(_graph);
        //_threads = new Dictionary<ulong, Thread>();
        _mfcc = new MelFrequencyCepstrumCoefficient(lowerFrequency: 20, 
            upperFrequency: 8000, windowLength: 0.04, frameRate: FrameRate, cepstrumCount: CeptrumCount, numberOfBins: 2048);
    }

    // Update is called once per frame
    void Update() {
        // Ring buffer
        var sampleLenght = (int)(Fs * Time.deltaTime);
        var data = new float[sampleLenght];
        _audioSource.GetOutputData(data, 0);
        for (int i = sampleLenght - 1; i > -1; i--) {
            _audioBuffer.Enqueue(data[i]);
        }

        if (_audioBuffer.Count > BufferLenght) {
            Predict();
        }
    }

    void Predict()
    {
        float maxData = -1;
        float[] dataArr = new float[BufferLenght];
        for (int i = 0; i < BufferLenght; i++) {
            float data = _audioBuffer.Dequeue();
            //tensorData[0, i, 0] = data;
            dataArr[i] = data;
            if (maxData < data) {
                maxData = data;
            }
        }
        
        if (maxData > 0.1) {
            var extractedFeatures = _mfcc.Transform(Signal.FromArray(dataArr, BufferLenght)).ToArray();
            var tensorData = new float[1, CeptrumCount, FrameRate, 1];
            for (int i = 1; i < CeptrumCount; i++) {
                for (int j = 0; j < FrameRate; j++) {
                    tensorData[0, i, j, 0] += (float)extractedFeatures[j].Descriptor[i];
                }
            }
            Thread newThread = new Thread(() => PredictAux(tensorData));
            //_threads.Add(_threadCounter, newThread);
            newThread.Start();
            //_threadCounter++;


            // Hacky code to record and extract features on the fly

            //string outputMatrix = "";
            //for (int i = 1; i < 13; i++) {
            //    for (int k = 0; k < 50; k++) {
            //        outputMatrix += extractedFeatures[k].Descriptor[i].ToString().Replace(',', '.');
            //        if (k != 49) {
            //            outputMatrix += ";";
            //        }
            //    }
            //    outputMatrix += "\n";
            //}
            //string rootFolder = @"D:\_Documents\#_Cours_TUT_2018_2019\Innovation Project\audio-recognition-game\sound_recognition\recorded_data\bag_";
            //File.WriteAllText(rootFolder + ccc++ + "_mic.csv", outputMatrix);
        }
    }

    void PredictAux(float[,,,] tensorData) 
    {
        var session = new TFSession(_graph);
        var runner = _session.GetRunner();
        TFTensor input = tensorData;
        runner.AddInput(_graph["input_node_input"][0], input);
        runner.Fetch(_graph["output_node/Sigmoid"][0]);

        var recurrentTensor = runner.Run()[0].GetValue() as float[,];

        // We dispose of resources so our graph doesn't break down over
        // time. IMPORTANT if you will repeatedly call the graph.
        //graph.Dispose();
        
        List<float> outputs = new List<float>();
        foreach (var v in recurrentTensor) {
            outputs.Add(v);
        }
        var max = outputs.Max();
        var idx = outputs.IndexOf(max);
        var action = _audioActions[idx];
        if (action == AudioAction.CLAP && max > 0.1 || action == AudioAction.KEYS && max > 0.15)
        {
            _actionController.QueueAction(action);
        } else if (outputs[2] > 0.0012)
        {
            _actionController.QueueAction(AudioAction.BAG);
        }
        else
        {
            //Debug.Log("Closest: " + action);
        }

        session.Dispose();
        //_threads.Remove(id);
    }
}

public enum AudioAction
{
    CLAP, KEYS, BAG
}