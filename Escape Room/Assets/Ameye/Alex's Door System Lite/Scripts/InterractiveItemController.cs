// Created by Alexander Ameye
// Modified by Thomas Foucault

using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;

public class InterractiveItemController : MonoBehaviour {

    // UI Settings
    public GameObject TextPrefab; // A text element to display when the player is in reach of the door
    [HideInInspector] public GameObject TextPrefabInstance; // A copy of the text prefab to prevent data corruption
    private List<Renderer> _illuminatedObjectRenderer;

    public GameObject CrosshairPrefab;
    [HideInInspector] public GameObject CrosshairPrefabInstance; // A copy of the crosshair prefab to prevent data corruption

    // Raycast Settings
    public float Reach = 4.0F;
    public Camera MainCamera;

    public Color DebugRayColor;

    private readonly string[] _interactibles = { "Door", "Bulb", "Note", "LightbulbHolder", "Switch", "Key", "LockedDoor" };
    private readonly Color _lighenColor = new Color(0.3f, 0.3f, 0.3f);

    public Transform NotePanel;
    private const string BookText = "- July, 3rd, 2005\n\nDear diary,\n\nToday, Duchess went hunting outside and brought back a mice! How wild she is.\n\nOh dear, her beautiful fur was so dirty I had to wash her in the sink. She hated it but I'm sure she knows it's for her own good.\n\nMy sweetheart was so terrified she jumped out of the sink and accidentaly pressed the hidden switch, oh my poor Duchess...";
    private const string NoteText = "ELECTROFIX - Electrical problem? We fix it!\n\n----------------------------------------------------------------------------------\n- Bathroom's neon replaced\t54$\n- Office investigation\t\t30$\n\nTotal:\t\t\t\t\t84$\n\nComments:\nThe clap recognition system in the office is broken, please contact the manufacturer";

    public List<GameObject> LightBulbsInHolder;
    private int _lightbulbGathered = 0;

    public Sprite BagSprite;
    public GameObject FrameSpriteObject;

    public SoundsManager SoundsPlayer;

    void Start() {
        _illuminatedObjectRenderer = new List<Renderer>();
    }

    void Update() {
        if (ActionController.GetBlocked()) {
            return;
        }

        // Set origin of ray to 'center of screen' and direction of ray to 'cameraview'
        Ray ray = MainCamera.ViewportPointToRay(new Vector3(0.5F, 0.5F, 0F));

        RaycastHit hit; // Variable reading information about the collider hit

        // Cast ray from center of the screen towards where the player is looking
        if (Physics.Raycast(ray, out hit, Reach)) {
            if (_interactibles.Contains(hit.collider.tag)) {
                // Illuminate Object
                var newRenderer = hit.transform.GetComponent<Renderer>();
                if (newRenderer == null) {
                    newRenderer = hit.transform.GetComponentInChildren<Renderer>();
                }
                if (!_illuminatedObjectRenderer.Contains(newRenderer)) {
                    DeIlluminate();
                }
                if (_illuminatedObjectRenderer.Count == 0 && (hit.collider.tag != "LightbulbHolder" || LightBulbsInHolder.Count > 0)) {
                    _illuminatedObjectRenderer.Add(newRenderer);
                    _illuminatedObjectRenderer.AddRange(hit.transform.GetComponentsInChildren<Renderer>());
                    foreach (var r in _illuminatedObjectRenderer) {
                        r.material.SetColor("_EmissionColor", _lighenColor);
                    }
                }
            }

            switch (hit.collider.tag) {
                case "Door":
                    DoorRotationLite dooropening = hit.transform.gameObject.GetComponent<DoorRotationLite>();

                    if (Input.GetButton("Action")) {
                        if (dooropening.RotationPending == false) StartCoroutine(dooropening.Move());
                    }
                    break;
                case "Note":
                    if (Input.GetButton("Action")) {
                        NotePanel.GetComponentInChildren<TextMeshProUGUI>().text = hit.collider.gameObject.name == "note" ? NoteText : BookText;
                        NotePanel.gameObject.SetActive(true);
                        ActionController.SetBlocked(true);
                    }
                    break;
                case "LightbulbHolder":
                    if (Input.GetButton("Action")) {
                        while (_lightbulbGathered > 0)
                        {
                            var l = LightBulbsInHolder.First();
                            if (l != null)
                                l.SetActive(true);
                            _lightbulbGathered--;
                            LightBulbsInHolder.RemoveAt(0);
                        }
                    }
                    break;
                case "Bulb":
                    if (Input.GetButton("Action")) {
                        hit.transform.gameObject.SetActive(false);
                        _lightbulbGathered++;
                    }
                    break;
                case "Switch":
                    if (Input.GetButton("Action"))
                    {
                        hit.transform.gameObject.tag = "Untagged";
                        hit.transform.gameObject.GetComponent<Animator>().SetTrigger("Press");
                        FrameSpriteObject.GetComponent<SpriteRenderer>().sprite = BagSprite;
                        ActionController.PressSwitch();
                        SoundsPlayer.PlayRipPaper();
                    }
                    break;
                case "Key":
                    if (Input.GetButton("Action")) {
                        hit.transform.gameObject.SetActive(false);
                        ActionController.PossessKey();
                    }
                    break;
                case "LockedDoor":
                    if (Input.GetButton("Action") && !hit.transform.gameObject.GetComponent<AudioSource>().isPlaying) {
                        hit.transform.gameObject.GetComponent<AudioSource>().Play();
                    }
                    break;
                default:
                    DeIlluminate();
                    break;
            }
        } else {
            DeIlluminate();
        }

        //Draw the ray as a colored line for debugging purposes.
        Debug.DrawRay(ray.origin, ray.direction * Reach, DebugRayColor);
    }

    void DeIlluminate() {
        foreach (var r in _illuminatedObjectRenderer) {
            r.material.SetColor("_EmissionColor", Color.black);
        }
        _illuminatedObjectRenderer.Clear();
    }
}
