#include<stdio.h>

int val(int, int);
int reff(*int, *int);
int main()
{
int a,b;
printf("Enter two numbers\na= ");
scanf("%d",&a);
printf("b= ");
scanf("%d",&b);
val(a,b);
reff(&a,&b);
return 0;
}

val(int a, int b)
{
int c=a+b;
printf("Addition by call by value: %d",c);
return 0;
}

reff(int *a,int *b)
{
int c=*a+*b;
printf("Addition by call by value: %d",c);
return 0;
}
