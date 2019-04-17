using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Scripts.Rooms {
    public class LivingRoom : RoomTrigger {
        void Start()
        {
            Room = Room.LIVINGROOM;
        }
    }
}
