using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Scripts.Rooms {
    public class Bathroom : RoomTrigger {
        void Start()
        {
            Room = Room.BATHROOM;
        }
    }
}
