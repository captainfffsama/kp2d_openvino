#include <iostream>
#include <string>
#include <memory>

using namespace std;

int main(int argc, char *argv[])
{
    auto_ptr<string> films[5]={
        auto_ptr<string> (new string("1")),
        auto_ptr<string> (new string("2")),
        auto_ptr<string> (new string("3")),
        auto_ptr<string> (new string("4")),
        auto_ptr<string> (new string("5"))
    };
    unique_ptr<string> pwin;
    pwin=films[2];
    for(int i=0;i<5;++i)
    {
        cout<< *films[i]<< endl;

    }
    cout<< *pwin <<endl;
    return 0;
}
