using System;
namespace HeapSortLinkedListOP
{
    class Program
    {
        static void Main()
        {
            int seed = (int)DateTime.Now.Ticks & 0x0000FFFF;                        //sukuriamas "seed" pagal kurį yra sugeneruojami atsitiktiniai skaičiai
            Test_Array_List(seed);                                                  //kviečiamas "Test_Array_List" kuris atlieka programai reikiamus veiksmus
            Console.ReadKey();
        }
        /// <summary>
        /// Sudaromas medis, kad šaknis būtų didesnė už vaikines šaknis.
        /// </summary>
        /// <param name="items">rūšiuojami simboliai</param>
        /// <param name="n">simbolių kiekis</param>
        /// <param name="i">indeksas</param>
        public static void Heapify(DataList items, int n, int i)
        {
            int largest = i;
            int l = 2 * i + 1;                                                      //šaknies kariės pusės vaikas
            int r = 2 * i + 2;                                                      //šaknies dešinės pusės vaikas
            if (l < n && items.GetValue(l) > items.GetValue(largest))               //ar kairės pusės vaikas didesnis už šaknį
                largest = l;
            if (r < n && items.GetValue(r) > items.GetValue(largest))               //ar dešinės pusės vaikas didesnis už šaknį
                largest = r;
            if (largest != i)                                                       //ar didžiausios reikšmės indeksas nesutampa su pradiniu didžiausios reikšmės indeksu
            {
                items.Swap(i,largest,items.GetValue(i), items.GetValue(largest));   //apkeičia šaknis, jeigu tenkina salygą
                Heapify(items, n, largest);                                         //kviečiamas metodas rekursiškai, jeigu tenkina salygą.
            }
        }
        /// <summary>
        /// aibės rūšiavimas "heap" metodu
        /// </summary>
        /// <param name="items">rūšiuojami simboliai</param>
        public static void HeapSort(DataList items)
        {
            int n = items.Length;                                                   //nustatomas aibės ilgis konsolėje iš klaviatūros
            for (int i = n / 2 - 1; i >= 0; i--)                                    //kviečiamas "Heapify" metodas n/2 - 1 kartų kad būtų paruoštas medis rikiavimui
                Heapify(items, n, i);
            for (int i = n-1; i >=0 ; i--)                                          //apkeičia šaknį su paskutiniu simboliu ir sumažina masyvo ilgį vienetu
            {
                items.Swap(0, i, items.Head(), items.GetValue(i));
                Heapify(items, i, 0);
            }
        }
        /// <summary>
        /// nustatoma kokio dydžio bus aibė, suskaičiuojamas laikas per kurį atliekami veiksmai, rezultatai išspausdinami į konsolės langą
        /// </summary>
        public static void Test_Array_List(int seed)
        {
            Console.WriteLine("HEAPSORT LINKED LIST OP");
            Console.WriteLine("--------------------------");
            Console.WriteLine("Iveskite duomenu kieki n");
            Console.Write("n = ");
            int n = int.Parse(Console.ReadLine());
            MyDataList mylist = new MyDataList(n, seed);
            Console.WriteLine("Pradiniai duomenys");
            mylist.Print(n);
            DateTime dateTime = DateTime.Now;
            HeapSort(mylist);
            Console.WriteLine();
            Console.WriteLine("Surusiuoti duomenys");
            mylist.Print(n);
            Console.WriteLine(DateTime.Now - dateTime);
        }
    }
}
