using System;
using System.Collections.Generic;
using System.Text;

namespace HeapSortLinkedListOP
{
    class MyDataList : DataList
    {
        class MyLinkedListNode
        {
            public MyLinkedListNode NextNode { get; set; }
            public int Data { get; set; }
            public MyLinkedListNode(int data)
            {
                this.Data = data;
            }
        }

        readonly MyLinkedListNode headNode;                             //pirmo simbolio mazgas
        MyLinkedListNode prevNode;                                      //ankstesnio simbolio mazgas
        MyLinkedListNode currentNode;                                   //dabartinio simbolio mazgas
        public MyDataList(int n, int seed)                              //sudaroma simbolių aibė
        {
            length = n;
            Random rand = new Random(seed);
            headNode = new MyLinkedListNode(rand.Next(0, 100));
            currentNode = headNode;
            for (int i = 1; i < length; i++)
            {
                prevNode = currentNode;
                currentNode.NextNode = new MyLinkedListNode(rand.Next(0, 100));
                currentNode = currentNode.NextNode;
            }
            currentNode.NextNode = null;
        }
        public override int Head()                                      //nustatoma kas yra pirmas simbolis "Head()"
        {
            currentNode = headNode;
            prevNode = null;
            return currentNode.Data;
        }
        public override int Next()                                      //nustatoma kas yra sekantis simbolis
        {
            prevNode = currentNode;
            currentNode = currentNode.NextNode;
            return currentNode.Data;
        }
        public override void Swap(int aIndex, int bIndex, int a, int b) //apkeičiami simboliai pagal indeksus
        {
            if (aIndex == 0)
            {
                Head();
                currentNode.Data = b;
                for (int i = 0; i < bIndex; i++)
                {
                    Next();
                }
                currentNode.Data = a;
            }
            else
            {
                Head();
                for (int i = 0; i < aIndex; i++)
                {
                    Next();
                }
                currentNode.Data = b;
                for (int i = aIndex; i < bIndex; i++)
                {
                    Next();
                }
                currentNode.Data = a;
                Head();
            }

        }
    }
}
