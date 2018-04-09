#include <random>
#include <algorithm>
#include <dirent.h>
#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<assert.h>
struct xyz{
    double x;
    double y;
    double z;
};
struct set
{
    int begin;
    int end;
    int size;
};
void READFILE(const std::string &dirfile,const std::string filename,std::vector<struct xyz>& re)
{
    std::fstream readfile;
    
    readfile.open(dirfile+"/"+filename,std::ios::in);
    std::cout<<dirfile+"/"+filename<<std::endl;
    if(!readfile)
    {
        
        std::cerr<<"open error"<<std::endl;
    }
    std::string line;
    while(readfile)
    {
        struct xyz element;
        //std::cout<<line<<std::endl;
        readfile>>element.x>>element.y>>element.z;
       // std::cout<<element.x<<element.y<<element.z<<std::endl;
        re.push_back(element);
    }
    readfile.close();
    //return re;
}
void READS(const std::string &filename,std::vector<struct set>& set1)
{
    std::fstream readfile;
    readfile.open(filename,std::ios::in);
    if(!readfile)
    {
        
        std::cerr<<"open error"<<std::endl;
    }
    while(readfile)
    {
        struct set element;

        readfile>>element.begin>>element.end>>element.size;
       // std::cout<<element.begin<<std::endl;
        set1.push_back(element);
    }
    readfile.close();
}
float* READPLOUD(const std::string &filename,int& num)
{
     std::fstream readfile;
    readfile.open(filename,std::ios::in);
    if(!readfile)
    {
        
        std::cerr<<"open error"<<std::endl;
    }
    int NUM=0;
    readfile>>NUM;
    num=NUM;
     
    float* data1=new float[NUM * 3];
    float*data2=data1;
  //  data=data1;
    while(readfile)
    {
       float x,y,z,label;
        readfile>>x>>y>>z>>label;
        *data1=x;
      //  std::cout<<*data<<std::endl;
        
        ++data1;
        *data1=y;
       // std::cout<<*data<<std::endl;
        

        ++data1;
        *data1=z;
     //   std::cout<<*data<<std::endl;
        
        ++data1;

    }
    readfile.close();
    return data2;
}
/*
int main()
{
    std::string namefile="t1.txt";
    std::vector<struct set> set;
    READS(namefile,set);
    int num=0;
    double *data=READPLOUD("t.txt",num); 
    return 0;
}*/