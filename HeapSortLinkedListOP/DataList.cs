using System;
using System.Collections.Generic;
using System.Text;

namespace HeapSortLinkedListOP
{
    /// <summary>
    /// sukuriama abstrakti "linkedlist" klasė
    /// </summary>
    abstract class DataList
    {
        protected int length;
        public int Length { get { return length; } }
        public abstract int Head();                                         //pirmas simbolis
        public abstract int Next();                                         //sekantis simbolis
        public abstract void Swap(int aIndex, int bIndex, int a, int b);    //simbolių apkeitimas
        public void Print(int n)                                            //spausdinimas
        {
            Console.Write(" {0,3} ", Head());
            for (int i = 1; i < n; i++)
                Console.Write(" {0,3} ", Next());
            Console.WriteLine();
        }
        public int GetValue(int index)                                      //paimama reikšmė tam tikro indekso
        {
            if (index == 0)
            {
                return Head();
            }
            else
            {
                Head();
                for (int i = 1; i < index; i++)
                {
                    Next();
                }
                return Next();
            }
        }
    }
}
