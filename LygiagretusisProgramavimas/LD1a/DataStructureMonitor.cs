using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace IFF_8_3_ZamasD_L1a
{
    class DataStructureMonitor
    {
        
        public DataStructure[] Data;
        private object locker;
        private static bool inUse;

        public DataStructureMonitor(int size)
        {
            Data = new DataStructure[size];
            locker = new object();
            inUse = false;
        }

        public void Put(DataStructure data, int place)
        {
            lock (locker)
            {
                while (inUse)
                {
                    Monitor.Wait(locker);
                }
                inUse = true;
                this.Data[place] = new DataStructure(data.Name, data.Year, data.Grade);
                Monitor.PulseAll(locker);
                inUse = false;
            }
        }

        public DataStructure Get(int place)
        {
            lock (locker)
            {
                while (inUse)
                {
                    Monitor.Wait(locker);
                }
                inUse = true;
                Monitor.PulseAll(locker);
                inUse = false;
                return this.Data[place];
            }
        }
    }
}
