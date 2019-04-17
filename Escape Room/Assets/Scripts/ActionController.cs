using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActionController : MonoBehaviour {
    public List<Light> LivingroomLights;
    public List<Light> BathroomLights;
    public List<Light> BedroomLights;
    public List<Light> OfficeLights;

    public Transform FallingBook;
    public Transform TVScreen;
    public GameObject NotePanel;

    public GameObject KeyTriggerArea;
    public GameObject LightbulbsTriggerArea;
    public List<GameObject> Lightbulbs;
    public GameObject Frame;
    public GameObject FrameHelpText;
    private bool _helpTextShown = false;

    private Room _room;
    private Dictionary<Room, List<Light>> _roomLights;

    private static bool _blocked = false;
    private static bool _frameChanged = false;

    private AudioAction? _pendingAction;
    private Collider _collider;
    private BoxCollider _keyAreaBoxCollider;
    private BoxCollider _lightbulbsAreaBoxCollider;

    public SoundsManager SoundsPlayer;
    public GameEventsManager GameManager;

    public GameObject KeyDoor;
    private AudioSource _keyDoorAudioSource;
    public AudioClip UnlockedSound;

    private static bool _possessKey;
    public GameObject TargetUI;


    // Start is called before the first frame update
    void Start() {
        _room = Room.LIVINGROOM;
        _roomLights = new Dictionary<Room, List<Light>>(4);
        _roomLights.Add(Room.LIVINGROOM, LivingroomLights);
        _roomLights.Add(Room.OFFICE, OfficeLights);
        _roomLights.Add(Room.BATHROOM, BathroomLights);
        _roomLights.Add(Room.BEDROOM, BedroomLights);

        _collider = transform.parent.GetComponent<CharacterController>().GetComponent<Collider>();
        _keyAreaBoxCollider = KeyTriggerArea.GetComponent<BoxCollider>();
        _lightbulbsAreaBoxCollider = LightbulbsTriggerArea.GetComponent<BoxCollider>();
        _keyDoorAudioSource = KeyDoor.GetComponent<AudioSource>();

    }

    // Update is called once per frame
    void Update() {
        if (!_blocked) {
            // Only for Debug? Or in case of mic doesn't work
            if (Input.GetButtonDown("Clap")) {
                DoAction(AudioAction.CLAP);
            }
            if (Input.GetButtonDown("Keys")) {
                DoAction(AudioAction.KEYS);
            }
            if (Input.GetButtonDown("Bag")) {
                DoAction(AudioAction.BAG);
            }
            if (_pendingAction != null) {
                DoAction(_pendingAction.Value);
                _pendingAction = null;
            }
            
            if (!TargetUI.activeSelf) {
                TargetUI.SetActive(true);
            }
        } else {
            if (TargetUI.activeSelf) {
                TargetUI.SetActive(false);
            }
        }

        if (Input.GetButtonDown("Cancel") && _blocked) {
            NotePanel.SetActive(false);
            SetBlocked(false);
        }

        if (_helpTextShown && FrameHelpText) {
            if (_room != Room.OFFICE) {
                FrameHelpText.SetActive(false);
            } else {
                FrameHelpText.SetActive(true);
            }
        }
    }

    public static void SetBlocked(bool b) {
        _blocked = b;
    }

    public static bool GetBlocked() {
        return _blocked;
    }

    public void DoAction(AudioAction action) {
        Debug.Log(action);
        switch (action) {
            case AudioAction.CLAP:
                foreach (var l in _roomLights[_room]) {
                    l.enabled = !l.enabled;
                    var r = l.transform.parent.GetComponent<Renderer>();
                    r.material.SetColor("_EmissionColor", r.material.GetColor("_EmissionColor") == Color.black ? Color.white : Color.black);
                }

                // First clap makes the book fall
                if (FallingBook && _room == Room.LIVINGROOM) {
                    FallingBook.GetComponent<Animator>().SetTrigger("Fall");
                    FallingBook.GetChild(0).GetChild(0).tag = "Note";
                    FallingBook = null;
                    SoundsPlayer.PlayBookFall();
                    TVScreen.GetChild(0).gameObject.SetActive(false);
                    TVScreen.GetComponent<Renderer>().material.SetColor("_EmissionColor", Color.black);
                }

                if (_frameChanged && _lightbulbsAreaBoxCollider.bounds.Intersects(_collider.bounds)) {
                    if (Frame == null) {
                        var mustSwitchOff = false;
                        foreach (var obj in Lightbulbs) {
                            if (obj.activeSelf) {
                                var l = obj.GetComponentInChildren<Light>();
                                l.enabled = !l.enabled;
                            } else {
                                mustSwitchOff = true;
                            }
                        }

                        if (mustSwitchOff) {
                            StartCoroutine(SwitchOffLightbulbs());
                        } else {
                            _keyDoorAudioSource.clip = UnlockedSound;
                            _keyDoorAudioSource.Play();
                            KeyDoor.tag = "Door";
                        }
                    } else if (!_helpTextShown && FrameHelpText) {
                        _helpTextShown = true;
                    }
                }
                break;
            case AudioAction.KEYS:
                if (_possessKey && _keyAreaBoxCollider.bounds.Intersects(_collider.bounds)) {
                    GameManager.Win();
                }
                break;
            case AudioAction.BAG:
                if (_frameChanged && Frame && _lightbulbsAreaBoxCollider.bounds.Intersects(_collider.bounds)) {
                    Frame.GetComponent<Animator>().SetTrigger("Fall");
                    FrameHelpText.SetActive(false);
                    FrameHelpText = null;
                    Frame = null;
                    SoundsPlayer.PlayFrameFall();
                }
                break;
        }
    }

    IEnumerator SwitchOffLightbulbs() {
        yield return new WaitForSeconds(1);
        foreach (var lightbulb in Lightbulbs) {
            lightbulb.GetComponentInChildren<Light>().enabled = false;
        }
    }

    public void QueueAction(AudioAction action) {
        if (!_blocked)
            _pendingAction = action;
    }

    public void ChangeRoom(Room room) {
        _room = room;
    }

    public static void PressSwitch() {
        _frameChanged = true;
    }

    public static void PossessKey() {
        _possessKey = true;
    }
}

public enum Room {
    LIVINGROOM, BEDROOM, BATHROOM, OFFICE
}