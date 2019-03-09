using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class GraphTest : MonoBehaviour
{
    public TextAsset graphModel;

    int[] inputTensor = new int[3];

    // Start is called before the first frame update
    void Start()
    {
        using (var graph = new TFGraph()) {

            graph.Import(graphModel.bytes);

            // We declare and run a session with our graph.
            var session = new TFSession(graph);
            var runner = session.GetRunner();

            // Here I implicitly convert my inputTensor (array) to 
            // TensorFlow tensor. The TFTensor will take on the dimensions
            // of the array.
            TFTensor input = new int[1,3]{ { 1 ,  2 , 3}};
            Debug.Log(graph["output_node/Sigmoid"]);
            // We tell the session to sub in our input tensor for our 
            // graph's placeholder tensor and fetch the predictions from the
            // output node.
            runner.AddInput(graph["input_node"][0], input);
            runner.Fetch(graph["output_node/Sigmoid"][0]);

            // We run the graph and store the probability of each result in 
            // recurrentTensor.
            var recurrentTensor = runner.Run()[0].GetValue() as float[,];

            // We dispose of resources so our graph doesn't break down over
            // time. IMPORTANT if you will repeatedly call the graph.
            session.Dispose();
            graph.Dispose();

            Debug.Log(recurrentTensor[0,3]);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
