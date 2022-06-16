using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace IFF_8_3_ZamasD_L1a
{
    class Program
    {
        static bool FirstWorking = false;
        static bool SecondWorking = false;
        static bool ThirdWorking = false;
        static bool FourthWorking = false;
        static void Main(string[] args)
        {
            string[] files = { "IFF_8_3_ZamasD_L1_dat_1.txt", "IFF_8_3_ZamasD_L1_dat_2.txt", "IFF_8_3_ZamasD_L1_dat_3.txt" };
            string[] saveFiles = { "IFF_8_3_ZamasD_L1_dat_1Result.txt", "IFF_8_3_ZamasD_L1_dat_2Result.txt", "IFF_8_3_ZamasD_L1_dat_3Result.txt" };

            for (int z = 0; z < files.Length; z++)
            {
                DataStructureMonitor dataStructure = ReadFile(files[z]);
                DataStructureMonitor initialData = ReadFile(files[z]);
                DataStructureMonitor newDataStucture = new DataStructureMonitor(dataStructure.Data.Length);
                Thread First = new Thread(() => NewValue(dataStructure.Get(0), 0, newDataStucture));
                First.Name = "First";
                First.Start();
                Thread Second = new Thread(() => NewValue(dataStructure.Get(0), 0, newDataStucture));
                Second.Name = "Second";
                Second.Start();
                Thread Third = new Thread(() => NewValue(dataStructure.Get(0), 0, newDataStucture));
                Third.Name = "Third";
                Third.Start();
                Thread Fourth = new Thread(() => NewValue(dataStructure.Get(0), 0, newDataStucture));
                Fourth.Name = "Fourth";
                Fourth.Start();
                for (int i = 0; i < dataStructure.Data.Length; i++)
                {
                    while (true)
                    {
                        FirstWorking = First.IsAlive;
                        SecondWorking = Second.IsAlive;
                        ThirdWorking = Third.IsAlive;
                        FourthWorking = Fourth.IsAlive;

                        if (FirstWorking == false)
                        {
                            int firstI = i;
                            First = new Thread(() => NewValue(dataStructure.Get(firstI), firstI, newDataStucture));
                            First.Name = "First";
                            First.Start();
                            break;
                        }

                        else if (SecondWorking == false)
                        {
                            int secondI = i;
                            Second = new Thread(() => NewValue(dataStructure.Get(secondI), secondI, newDataStucture));
                            Second.Name = "Second";
                            Second.Start();
                            break;
                        }

                        else if (ThirdWorking == false)
                        {
                            int thirdI = i;
                            Third = new Thread(() => NewValue(dataStructure.Get(thirdI), thirdI, newDataStucture));
                            Third.Name = "Third";
                            Third.Start();
                            break;
                        }

                        else if (FourthWorking == false)
                        {
                            int FourthI = i;
                            Fourth = new Thread(() => NewValue(dataStructure.Get(FourthI), FourthI, newDataStucture));
                            Fourth.Name = "Fourth";
                            Fourth.Start();
                            break;
                        }
                    }
                }
                if (First.IsAlive) { First.Join(); }
                if (Second.IsAlive) { Second.Join(); }
                if (Third.IsAlive) { Third.Join(); }
                if (Fourth.IsAlive) { Fourth.Join(); }
                WriteFile(newDataStucture.Data, initialData.Data, saveFiles[z]);
            }
            Console.ReadKey();

        }
        public static DataStructureMonitor ReadFile(string fileName)
        {
            try
            {
                using (StreamReader reader = new StreamReader(File.Open(fileName, FileMode.Open)))
                {
                    List<DataStructure> tempList = new List<DataStructure>();
                    string[] splitReadLine;

                    while (reader.EndOfStream == false)
                    {
                        splitReadLine = reader.ReadLine().Split(';');
                        tempList.Add(new DataStructure(splitReadLine[0], int.Parse(splitReadLine[1]), Double.Parse(splitReadLine[2])));
                    }

                    DataStructureMonitor fileLines = new DataStructureMonitor(tempList.Count);
                    int itteration = 0;
                    foreach (DataStructure data in tempList)
                    {
                        fileLines.Put(new DataStructure(data), itteration);
                        itteration++;
                    }
                    return fileLines;
                }
            }
            catch (IOException e)
            {
                Console.WriteLine("The file could not be read:");
                Console.WriteLine(e.Message);
                return null;
            }
        }
        public static DataStructure Filter(DataStructure data)
        {
            DataStructure tempStructure = null;
            if (Char.IsLetter(data.Name[0]))
            {
                tempStructure = new DataStructure(data.Name, data.Year, data.Grade);
            }
            return tempStructure;
        }
        public static void NewValue(DataStructure data, int place, DataStructureMonitor newDataStructure)
        {
            double unroundedCezarValue = data.Year * data.Grade;
            int roundedCezarValue = Convert.ToInt32(unroundedCezarValue);
            string newValue = "";
            foreach (char c in data.Name)
            {
                newValue += (char)(c + roundedCezarValue);
            }
            Console.WriteLine(data.Name + " " + newValue + " " + Thread.CurrentThread.Name);
            if (Filter(new DataStructure(newValue, data.Year, data.Grade)) != null)
            {
                newDataStructure.Put(new DataStructure(newValue, data.Year, data.Grade), place);
            }
        }
        public static void WriteFile(DataStructure[] endData, DataStructure[] startingData, string saveFile)
        {
            try
            {
                if (File.Exists(saveFile))
                {
                    File.Delete(saveFile);
                }

                using (StreamWriter writer = new StreamWriter(File.Open(saveFile, FileMode.OpenOrCreate)))
                {
                    writer.WriteLine("Initial Information");
                    foreach (DataStructure line in startingData)
                    {
                        writer.WriteLine(line.Name + " " + line.Year + " " + line.Grade);
                    }
                    writer.WriteLine("----------------------------------------------------------------------------");
                    writer.WriteLine("Encrypted Information");
                    foreach (DataStructure line in endData)
                    {
                        if (line == null)
                        {
                            break;
                        }
                        writer.WriteLine(line.Name + " " + line.Year + " " + line.Grade);
                    }
                }
            }
            catch (IOException e)
            {
                Console.WriteLine("The file could not be read:");
                Console.WriteLine(e.Message);
            }
        }
    }
}
