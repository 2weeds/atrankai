using System;
using System.Collections.Generic;
using System.Text;

namespace IFF_8_3_ZamasD_L1a
{
    class DataStructure
    {
        public string Name { private set; get; }
        public int Year { private set; get; }
        public double Grade { private set; get; }
        public DataStructure()
        {

        }
        public DataStructure(string name, int year, double grade)
        {
            this.Name = name;
            this.Year = year;
            this.Grade = grade;
        }
        public DataStructure(DataStructure data)
        {
            this.Name = data.Name;
            this.Year = data.Year;
            this.Grade = data.Grade;
        }
    }
}
