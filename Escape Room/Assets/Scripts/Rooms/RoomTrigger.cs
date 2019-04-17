using UnityEngine;

namespace Assets.Scripts.Rooms {
    public class RoomTrigger : MonoBehaviour
    {
        protected Room Room;

        void OnTriggerEnter(Collider col) {
            if (col.gameObject.tag == "Player") {
                col.gameObject.GetComponentInChildren<ActionController>().ChangeRoom(Room);
            }
        }
    }
}
