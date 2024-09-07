#include<stdio.h>

int main()
{
struct data
{
int roll, mrks[3], gr;
char name[20], dept[20];
}stud[5];
int n,i;

for (i=0;i<5;i++)
{
printf("Enter the G.R. Number of student: ");
scanf("%d",&stud[i].gr);
printf("Enter the Name of the student: ");
gets(stud[i].name);
printf("Enter the Department name: ");
gets(stud[i].dept);
printf("Enter the Roll Number: ");
scanf("%d",&stud[i].roll);
printf("Enter the marks in C.P.: ");
scanf("%d",&stud[i].mrks[0]);
printf("Enter the marks in A.P: ");
scanf("%d",&stud[i].mrks[1]);
printf("Enter the marks in Calculus: ");
scanf("%d",&stud[i].mrks[2]);
}

printf("Enter the G.R. Number of the student whose record is to be printed: ");
scanf("%d",&n);
for (i=0;i<67;i++)
{
if(n==stud[i].gr)
{
printf("Name: %s \nDepartment: %s\nRoll number: %d\nMarks in C.P.: %d\nMarks ib A.P.:%d\nMarks in Calculus: %d",stud[i].name,stud[i].dept,stud[i].roll,stud[i].mrks[0],stud[i].mrks[1],stud[i].mrks[2]);
break;
}
}
return 0;
}

