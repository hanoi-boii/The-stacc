#include<stdio.h>

void fibonacci(int);
void odd(int);
void even(int);
void prime(int);
int main()
{ char opt;
choice:
	int x,n;
	printf("\n1.fibonacci series\n");
	printf("2.odd series\n");
	printf("3.even series\n");
	printf("4.prime series\n");
	scanf("%d",&x);
	printf("how many terms you want? ");
	scanf("%d",&n);
	switch(x)
	{
		case 1:
		fibonacci(n-1);
		break;
		case 2:
		odd(n);
		break;
		case 3:
		even(n);
		break;
		case 4:
		prime(n);
		break;
		default :
		printf("Invalid Input");
	}
	printf("\nWant to continue? (y/n)");
	fflush(stdin);
	scanf("%c",&opt);
	if(opt=='y')
	goto choice;

return 0;
}

void fibonacci(int n)
{
	int f=0,s=1,sum=0;
	printf("0");
	for(int i=1;i<n+1;i++)
	{
		printf("\t%d",sum);
		f=s;
		s=sum;
		sum=f+s;
	}
}
void odd(int n)
{
	for(int i=1;i<2*n;i=i+2)
		printf("%d\t",i);
}
void even(int n)
{
	for(int i=0;i<2*n;i=i+2)
		printf("%d\t",i);
}
void prime(int n)
{
	int count=0;
		for(int i=1; count!=n;i++)
       {
         int fact=0;
        for(int j=1; j<=i; j++)
        {
            if(i%j==0)
              {  fact++;

              }
        }
        if(fact==2)
        {
        	count++;
            printf("%d " ,i);
        }
    }
}
