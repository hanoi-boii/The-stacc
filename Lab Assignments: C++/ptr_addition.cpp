#include<stdio.h>

int main()
{
int a[10], *ptr, sum=0;
printf("Enter 10 elements in the array ");
for(int i=0;i<10;i++)
scanf("%d",&a[i]);
ptr=&a[0];
for(int i=0;i<10;i++)
{sum=sum+*ptr;
ptr++;
}
printf("Sum is %d",sum);
return 0;
}
