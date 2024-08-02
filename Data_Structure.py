def array_list():
    print('''\n
    array=[0,1,2,3,4,5]
    newArray=[None]*len(array)

    Iteration:
    for i in range(len(array)):
        #copyArray
        newArray[i]=array[i]
        print(array[i], end=' ')
    Reverse iteration:
    for i in range(len(array)-1,-1,-1):
        print(array[i], end=' ')
    
    reverse out of place-------
    revArray=[None]*len(array)
    i=0
    j=len(array)-1
    while i<len(array):
        revArray[i]=array[j]
        i+=1
        j-=1
    print(revArray)
          
    reverse in place-----------
    while i<j:
        temp=array[i]
        array[i]=array[j]
        array[j]=temp
        i+=1
        j-=1
    print(revArray)
    shift left or right---------
    array[i-1]=array[i] or array[i]=array[i-1]
          
    rotating left or right------
    temp=array[0] then same as shifting 
    and put array[len(array)-1]=temp at last
    Removing and inserting is same as creating temp

    Fixed Capacity/Single Data type/Removal and insertion difficult/Random Access better than linked list

    Example--------
    Given a sorted array arr of distinct integers. 
    Sort the array into a wave-like array(In Place).
    In other words, arrange the elements into a sequence 
    such that arr[1] >= arr[2] <= arr[3] >=arr[4] <= arr[5]

    arr=[1,2,3,4,5,6,7]
    new=[]
    for i in range(len(arr)-1):
        if i%2==0:
            temp, arr[i]=arr[i], arr[i+1]
            arr[i+1]=temp
            new.append(arr[i])
            new.append(arr[i+1])
    if len(arr)%2!=0:
        new.append(arr[len(arr)-1])
    print(arr)#inplace
    print(new)#outplace 
    #[2, 1, 4, 3, 6, 5, 7]''')

#-----------------------------------------------------------------------------


def multidimentional_array():
    print('''\n
    Multidimensional array to linear array as RAM stores linear array memory
    The dimension of the given array is [4][4][8] and 
    length is 128. We need to map the dimension of linear index 111.
          
    X * (4*8) + Y*8 + Z = 111 where 0 <= X <= 3, 0 <= Y <= 3 and 0 <= Z <= 7.
    X = 111//(4*8) = 3 and 111 % (4*8) = 15
    Y = 15 // 8 = 1 and 15 % 8 = 7
    Z = 7
    So, the dimension of linear index 111 is [3][1][7].
    import numpy as np
    arr=np.zeros((2,2), dtype=int)
    arr2=[[1,2],
          [3,4]]
    row, col=arr.shape
    row2, col2=2,2
    result=np.zeros((row,col),dtype=int)

    Addition matrix:
    if row==row2 and col==col2:
        for i in range(row):
            for j in range(col2):
                result[i][j]=arr[i][j]+arr2[i][j]
                #Multiplication matrix:
                for k in range(col):
                    result[i][j]+=arr[i][k]*arr2[k][j]
  
    sum=0
    sum_of_row=np.zeros((row,1), dtype=int)
    sum_of_col=np.zeros((1,col),dtype=int)
    sum_of_digonal=0

    for i in range(row):
         for j in range(col):
             arr[i][j]=input() #create the 2D array

    for i in range(row):
        #swapping
        arr[i][0], arr[i][1]=arr[i][1], arr[i][0]
        sum_of_digonal+=arr[i][i]

        for j in range(col):
            sum+=arr[i][j]
            sum_of_row[i][0]+=arr[i][j] 
            sum_of_col[0][i]+=arr[i][j] #i,j and row, col in loop will change            

            print(arr[i][j], end=' ')   
        print() #iteration row wise''')


#-----------------------------------------------------------------------------


def singly_linked_list():
    print('''
    class Node:
        def __init__(self, elem, next):
            self.elem=elem
            self.next=next
    arr=[1,2,3,4,5]

    #creating link-----------
    head=Node(arr[0], None)
    tail=head
    for i in range(1, len(arr)):
          n=Node(arr[i], None)
          tail.next=n
          tail=tail.next

    count=0 #count can be treated as index
    element=None
    node=None
    node2=None
    node3=None
    idx=3 #the one you are finding
    temp=head
    while temp!=None:
        if count==idx:
            element=temp.elem #searching
            node=temp
        if count==idx-1:
            node2=temp
        if count==len(arr):
             node3=temp
        count+=1
        print(temp.elem, end=' ')
        temp=temp.next

    #insertion
    elem=10
    if idx==0:
        n=Node(elem, head)
        head=n
    elif idx>=1 and idx<count:
        n=Node(elem,head)
        n1=node2
        n2=node
        n.next=n2
        n1.next=n
    elif idx==count:
        n=Node(elem, None)
        n1=node3
        n1.next=n
    else:
        print('invalid')

    #Removal
    if idx==0:
        head=head.next
    elif idx>=1 and idx<count:
        n1=node2
        removed=n1.next
        n1.next=removed.next

    #reversing out of place
    newHead=Node(head.elem, None)
    temp=head.next
    while temp!=None:
        n=Node(temp.elem,newHead)
        newHead=n
        temp=temp.next
    temp=newHead
    while temp!=None:
        print(temp.elem, end=' ')
        temp=temp.next

    #reverse in place
    newHead2=None
    temp=head
    while temp!=None:
        n=temp.next
        temp.next=newHead2 
        newHead=temp 
        temp=n

    #rotating left
    new=head.next
    temp=new
    while temp.next!=None:
        temp=temp.next
    temp.next=head
    head.next=None
    head=new

    #rotating right
    temp=head.next
    temp2=head
    while temp.next!=None:
        temp=temp.next #last
        temp2=temp2.next #sec last
    temp2.next=None
    temp.next=head
    head=temp
          
    Given a Linked List of integers, write a function to modify the linked list such that all even
    numbers appear before all the odd numbers in the modified linked list. Also, keep the order of
    even and odd numbers the same.
    Examples:
    Input: 17 → 15 → 8 → 12 → 10 → 5 → 4 → 1 → 7 → 6 → None
    Output: 8 → 12 → 10 → 4 → 6 → 17 → 15 → 5 → 1 → 7 → None
     
    arr=[5,7,6,9,8,10]

    if arr[0]%2==0:
        head=Node(arr[0], None)
        tail=head
        for i in range(1, len(arr)):
            if arr[i]%2==0:
                n=Node(arr[i], None)
                tail.next=n
                tail=tail.next
    else:
        head=Node(arr[0], None)
        tail=head
        for i in range(1, len(arr)):
            if arr[i]%2!=0:
                n=Node(arr[i], None)
                tail.next=n
                tail=tail.next
    head2=None
    temp=head.next
    while temp.next!=None:
         temp=temp.next
    head2=temp

    tail=head2
    for i in range(len(arr)):
        if arr[0]%2==0:
            if arr[i]%2!=0:
                n=Node(arr[i], None)
                tail.next=n
                tail=tail.next
        else:
            if arr[i]%2==0:
                n=Node(arr[i], None)
                tail.next=n
                tail=tail.next

    temp1=head
    while temp1!=None:
         print(temp1.elem, end=' ')
         temp1=temp1.next      
          
          ''')
    

#-----------------------------------------------------------------------------


def doubly_linked_list():
    print('''
    class Node:
        def __init__(self, elem, next, prev):
            self.elem=elem
            self.next=next
            self.prev=prev
    
    arr=[1,2,3,4,5]
    
    def createlist(arr):
        dh=Node(None, None, None)
        dh.next=dh
        dh.prev=dh
        tail=dh
        for i in range(len(arr)):
            n=Node(arr[i],dh,tail)
            tail.next=n
            tail=tail.next
            dh.prev=tail
        return dh
    
    def iteration(dh):
        temp=dh.next
        while temp!=dh:
            print(temp.elem, end=' ')
            temp=temp.next
    
    def nodeAt(dh, idx):
        temp=dh.next
        c=0
        while temp!=dh:
            if c==idx:
                return temp
            temp=temp.next
            c+=1 
    
    def insertion(dh, idx):
        elem=10
        node_to_insert=Node(elem, None, None)
        node_of_index=nodeAt(dh, idx)
        prev_node=node_of_index.prev
    
        node_to_insert.next=node_of_index
        node_to_insert.prev=prev_node
        prev_node.next=node_to_insert
        node_of_index.prev=node_to_insert
        
    dh=createlist(arr)
    insertion(dh,3)
    
    def removal(dh, idx):
        node_to_remove=nodeAt(dh,idx)
        prev_node=node_to_remove.prev
        next_node=node_to_remove.next
    
        prev_node.next=next_node
        next_node.prev=prev_node
        node_to_remove.next=None
        node_to_remove.prev=None
    removal(dh, 3)
    
    #Given a dummy headed doubly circular linked list, rotate it 
    # left by k node (where k is a positive integer)
    
    def rotateleft(dh, idx):
        node_to_remove=nodeAt(dh,idx)
        prev_node=node_to_remove.prev
        next_node=node_to_remove.next
    
        prev_node.next=next_node
        next_node.prev=prev_node
        node_to_remove.next=None
        node_to_remove.prev=None
    
        temp=dh.next
        while temp.next!=dh:
            temp=temp.next
        temp.next=node_to_remove
        node_to_remove.prev=temp
        node_to_remove.next=dh
        dh.prev=node_to_remove
    
    idx=3
    for i in range(idx):
        rotateleft(dh, 0)

    iteration(dh)''')

#-----------------------------------------------------------------------------


def stack():
    print('''
    #An application is an "undo" mechanism in text editors; this operation is accomplished by keeping all text changes in 
    a stack. Popping the stack is equivalent to "undoing" the last action. A very similar one is going back pages in a 
    browser using the back button; this is accomplished by keeping the pages visited in a stack. Last in First Out 
    Rule=LIFO
    class Node:
        def __init__(self, elem, next):
            self.elem=elem
            self.next=next
    class Stack:
        def __init__(self, elem):
            self.top=None
        def push(self, elem):
            if self.top==None:
                self.top=Node(elem, None)
            else:
                newNode=Node(elem,None)
                newNode.next=self.top
                self.top=newNode
        def pop(self, elem):
            if self.top==None:
                return None
            else:
                popped=self.top
                self.top=self.top.next
                popped.next=None
                return popped.elem
        def peek(self):
            if self.top==None:
                return None
            else:
                return self.top.elem
            
    def checkpalindrome():
        d=Stack(None)
        inp='madam'
        inp2=''
        for i in inp:
            d.push(i)
        for i in range(len(inp)):
            inp2+=d.pop(i)
        if inp==inp2:
            return "it's a palindrome"
        else:
            return 'not a palindrome'
    
    checkpalindrome()
              ''')

#-----------------------------------------------------------------------------


def queue():
    print('''
    #First In First Out (FIFO) when we execute 100 
    programs in our 8-core CPU, the processor also 
    maintains a queue for command execution (Though 
    the scenario is a bit more complex which you will 
    learn in Operating Systems).
    class Node:
        def __init__(self, elem, next):
            self.elem=elem
            self.next=next
    class Queue:
        def __init__(self):
            self.front=None
            self.back=None
        def enqueue(self, elem):
            if self.front==None:
                self.front=Node(elem, None)
                self.back=self.front
            else:
                newNode=Node(elem,None)
                self.back.next=newNode
                self.back=self.back.next
        def dequeue(self, elem):
            if self.front==None:
                return None
            else:
                popped=self.front
                self.front=self.front.next
                return popped.elem
        def peek(self):
            if self.front==None:
                return None
            else:
                return self.front.elem
    
    #One reason for their importance is that 
    scenarios where you need a linked list behave 
    like a stack or queue is very frequent. For 
    example, you need stacks for language parsing, 
    expression evaluation, function calls, efficient 
    graph traversals, etc. Queues are similarly 
    important for resource scheduling, time sharing 
    of CPU among candidate processes, message routing 
    in network switches and routers, and efficient 
    graph traversal.
          ''')

#-----------------------------------------------------------------------------


def recursion():
    print('''
    #Recursive function is no different than a normal 
    function. The motivation to use recursive 
    functions vs non-recursive is that the recursive 
    solutions are usually easier to read and 
    comprehend. Certain applications, like tree 
    search, directory traversing etc. 
    def fact(n):
        if n==0 or n==1:
            return 1
        else:
            return n*fact(n-1)
    print(fact(5))
    
    def recursivesum(head):
        if head.next==None:
            return head.elem
        else:
            return head.elem+recursivesum(head.next)
    
    def strlength(str1):
        if len(str1)==0:
            return 0
        else:
            return 1+strlength(str1[1:])
        
    def listlength(head):
        if head==None:
            return 0
        else:
            return 1+listlength(head.next)
    
    def contains(head, key):
        if head==None:
            return None
        elif head==key:
            return head.elem
        else:
            return contains(head.next, key)
        
    def contain(arr, left, right, key):
        left=0
        right=len(arr)-1
        if left>right:
            return False
        else:
            mid=left+right//2
            if key>arr[mid]:
                return contain(arr,mid+1,right,key)
            else:
                contain(arr, left, mid-1, key)
    
    def maximum(a,b):
        return a if a>=b else b
    def findmax(arr, left):
        if left==len(arr)-1:
            return arr[left]
        else:
            maxRest=findmax(arr, left+1)
            return maximum(arr[left],maxRest)
        
    def findminNode(head):
        if head.next==None:
            return head
        else:
            minNode=head
            minNodeRest=findminNode(head.next)
            if minNodeRest.elem<head.elem:
                minNode=minNodeRest
            return minNode
        
    def selectSort(arr,left):
        if left==len(arr)-1:
            return
        else:
            minIdx=findMinIdx(arr, left, len(arr)-1)
            swap(arr, left, minIdx)
            selectSort(arr, left+1)
    def swap(arr, a, b):
        temp=arr[a]
        arr[a]=arr[b]
        arr[b]=temp
    def findMinIdx(arr, left, right):
        if left==right:
            return left
        else:
            mid=(left+right)//2
            minIdxLeft=findMinIdx(arr, left, mid)
            minIdxRight=findMinIdx(arr, mid+1, right)
            minIdx=minIdxLeft
            if arr[minIdxRight]<arr[minIdxLeft]:
                minIdx=minIdxRight
            return minIdx
        
    def insertSort(arr, left, right):
        if left>=right:
            return
        else:
            insertSort(arr, left, right-1)
            key=arr[right]
            j=right-1
            while j>=0 and key<arr[j]:
                arr[j+1]=arr[j]
                j-+1
            arr[j+1]=key
    
    def exp(a, n):
        if n==0:
            return 1
        else:
            return a*exp(a, n-1)
    
    def exp(a, n):
        if n==0:
            return 1
        elif n%2==0:
            return exp(a, n/2)*exp(a, n/2)
        else:
            return exp(a, (n-1)/2)*exp(a, (n-1)/2)*a
        
    #Memoization
    
    def fib(n):
        f=[-1]*(n+1)
        return M_fib(n, f)
    def M_fib(n, f):
        if n<2:
            return n
        else:
            if f[n]==-1:
                f[n]=M_fib(n-1,f)+M_fib(n-2, f)
            return f[n]
    # Take 123 as a int and add them all like 1+2+3:
    def digitsum(n):
        if n<=0:
            return 0
        else:
            return int(n%10)+digitsum(n/10)
    print(digitsum(123))
    
          ''')

#-----------------------------------------------------------------------------


def hashing():
    print('''
    One very important aspect is that in both array and linked list, our data is stored against some index value for our 
    using purpose.Another important concept of storing data is using key-value pair to store data instead of storing 
    against an index. The idea is to find the data in constant time using that key. You have already seen this type of 
    data structure in the Python dictionary.
    # 
    The idea of hashing and hash table data structure comes to solve this issue. The idea of hashing is that there will 
    be a function that will map any key as an integer, that integer will be used as an index of an array, and both key 
    and value will be stored in that index. It will be stored in a hash table with length 5 (which means the array’s 
    length is 5). Now the hash function will convert the string “name” and map it to an integer. E.g.: using summation 
    ASCII value of “name” and then mod it by 5 (in order to get a valid index). We get the index (110 + 97 + 109 + 101) = 
    417 % 5 = 2
    # 
    Hashing is best for collision handling
    # 
    Hash Function: hash(key) = key % 5
    
    Now, let's insert some key-value pairs into the hash table:
    
    Insert (Key: 12, Value: "Apple")
    Insert (Key: 5, Value: "Orange")
    Insert (Key: 17, Value: "Banana")
    Insert (Key: 10, Value: "Grapes")
    Insert (Key: 22, Value: "Watermelon")
    Insert (Key: 15, Value: "Pineapple")
    Here's the resulting hash table after each insertion:
    
    Step 1: Insert (Key: 12, Value: "Apple") index 2
    
    Index	Key-Value Pair
    2	(12, "Apple")
    
    Step 2: Insert (Key: 5, Value: "Orange") index 0
    
    Index	Key-Value Pair
    2	(12, "Apple")
    0	(5, "Orange")
    
    Step 3: Insert (Key: 17, Value: "Banana") Collision at index 2
    
    Index	Key-Value Pair
    2	(17, "Banana") -> (12, "Apple") #Forward chaining with collision
    0	(5, "Orange")
    	
    
    Step 4: Insert (Key: 10, Value: "Grapes") Collision at index 0
    
    Index	Key-Value Pair
    2	(17, "Banana") -> (12, "Apple")
    0	(10, "Grapes") -> (5, "Orange")
    	
    	
    
    Step 5: Insert (Key: 22, Value: "Watermelon") Collision at index 2
    
    Index	Key-Value Pair
    2	(22, "Watermelon") -> (17, "Banana") -> (12, "Apple")
    0	(10, "Grapes") -> (5, "Orange")
    	
    	
    
    Step 6: Insert (Key: 15, Value: "Pineapple") - Collision at index 0
    
    Index	Key-Value Pair
    2	(22, "Watermelon") -> (17, "Banana") -> (12, "Apple")
    0	(15, "Pineapple") -> (10, "Grapes") -> (5, "Orange")
          

    class Node:
      def __init__(self, key, val):
        self.key=key
        self.val=val
        self.next=None
    def insert(keys, values, hash):#insertion
      i=0
      while i<len(keys):
        index=hasF(keys[i])
        if hash[index]==None:
          n=Node(keys[i],values[i])
          hash[index]=n
        else:
          h=hash[index]
          while h.next!=None:
            h=h.next
          h.next=Node(keys[i], values[i])
        i=i+1
    def hasF(keys):#hashfunction
      sum=0
      for x in keys:
        sum=sum+int(x)
      return sum%5
    def printH(hash):#printing
      i=0
      while i<len(hash):
        j=hash[i]
        print(str(i)+"-->", end="")
        while j!=None:
          print(j.key, end="_")
          print(j.val, end=' ')
          j=j.next
        print()
        i=i+1
    keys=["121", "32", "56", "23"]
    values= ["lee", "Akram", "Ambrose", "Bond"]
    
    hash=[None]*5
    insert(keys, values, hash)
    printH(hash)
          ''')

#-----------------------------------------------------------------------------


def tree():
    print('''
    Terminology:
    
    Root/Leaf/None-Leaf Node
    Each tree consists of one or more nodes that are interlinked. The topmost node is called the root node.
    
    Parent/Child
    The direction of the tree is from top to bottom. Which means Node B is the immediate successor of node A, and Node A is the immediate predecessor of node B.
    
    Siblings
    All the nodes having the same parent are known as siblings. Therefore, B, C, and D are siblings, and F and G are siblings.
    
    Edge/Path
    The link between two nodes is called an edge.
    
    Degree
    The number of children of a node is known as degree.
          
    Depth
    The length of the path from a node to the root node is known as the depth.
    
    Height
    The length of the longest path from a node to one of its leaf nodes is known as the height.
    
    Level
    Each hierarchy starting from the root is known as the level.
    
    Subtree
    A tree that is the child and childs child of a Node.
    
    Characteristics of a Tree
    A tree must be continuous and not disjoint, which means every single node must be traversable starting from the root node. A tree cannot have a cycle. This means, for a tree, no of edges == no of nodes - 1.
    
    Tree Coding(using Linked List)
    Binary Trees:(N-ary Tress(N=no. of child))
    
    Parent=i/parent=i/2(if 1-n)
    left=2i, right=2i+1(if 1-n)
    Parent=i/parent=i/2(if 0-n)
    left=2i+1, right=2i+2(if 0-n)
    The maximum number of nodes at level i is: 2i
          
    Characteristics of a binary tree
    Binary Tree Traversal:(Draw the SLR's on the tree while traversing)
    Pre-order[SLR]
    In-order[LSR]
    Post-order[LRS]
          
    Types of Binary Tree:
    Full Binary Tree(all parent has 2 childs or 0 child)
    Complete Binary Tree(they arrange themselves serially starting from the left side)
    Perfect Binary Tree(complete and full)
    Balanced Binary Tree(the height of the left and right subtree of any node differ by not more than 1)
          
    Binary Tree Coding:
    Tree Contruction using Array
    Level, Height and Depth finding
    Numbers of Nodes Finding
    Identifying Tree Types: Full Complete Perfect

               
    Binary Search Tree:
    Characteristics of a BST
    Basic Operations:
    Creation
    Inserting a Node
    Removing a Node
    Searching For an element
    BST Traversal: Pre-order, In-order, Post-order
    Balancing BST
    Balanced vs unbalanced BST
          
    class TreeNode:
      def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def count_node(root):
      if root is None:
        return 0
      else:
        return 1 +count_node(root.left)+count_node(root.right)
    
    def height(root):
      if root is None:
        return -1
      else:
        left_Node=height(root.left)
        right_Node=height(root.right)
        if left_Node>right_Node:
          return left_Node+1
        else:
          return right_Node+1
    
    def get_level(node, node_lvl, elem):#also works for depth finding
      if node==None:
        return 0
      if node.value==elem:
        return node_lvl
      downlvl=get_level(node.left, node_lvl+1, elem)
      if downlvl!=0:
        return downlvl
      downlvl=get_level(node.right, node_lvl+1, elem)
      return downlvl
    
    def print_inorder(root):#In-order[LSR]
      if root:
        print_inorder(root.left)
        print(root.value, end=' ')
        print_inorder(root.right)
    
    def pre_order(root):#Pre-order[SLR]
      if root:
        print(root.value, end=' ')
        print_inorder(root.left)
        print_inorder(root.right)
    
    def post_order(root):#Post-order[LRS]
      if root:
        print_inorder(root.left)
        print_inorder(root.right)
        print(root.value, end=' ')
    
    def tree_contruction(arr, i):
      root=None
      if i<len(array):
        if arr[i]!=None:
          root=TreeNode(arr[i])
          root.left=tree_contruction(arr, 2*i)
          root.right=tree_contruction(arr, 2*i+1)
      return root
    
    array_rep=[None]*16#root height is 3 so (3+1)**2
    def array_contruction(arr, i):
      if arr==None:
        return None
      else:
        array_rep[i]=arr.value
        array_contruction(arr.left, 2*i)
        array_contruction(arr.right, 2*i+1)
      return array_rep
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)
    root.left.left.left = TreeNode(8)
    root.left.left.right = TreeNode(9)
    
    def is_full_tree(root):
      if root==None:
        return True
      if root.left is None and root.right is None:
        return True
      if root.left and root.right:
        return is_full_tree(root.left) and is_full_tree(root.right)
      else:
        return False
    
    def is_complete(root, index, numberNodes):
      if root is None:
        return True
      if index>=numberNodes:
        return False
      return is_complete(root.left, 2*index+1, numberNodes) and is_complete(root.right, 2*index+2, numberNodes)
    
    def depth(node):
      d=0
      while node is not None:
        d+=1
        node=node.left
      return d
    def is_perfect(root, d, level=0):
      if root is None:
        return True
      if root.left is None and root.right is None:
        return (d==level+1)
      if root.left is None or root.right is None:
        return False
      return is_perfect(root.left, d, level+1) and is_perfect(root.right, d, level+1)
    
    def isSymmetric(root):
        def isMirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.value == right.value) and isMirror(left.left, right.right) and isMirror(left.right, right.left)
    
        if not root:
            return True
        return isMirror(root.left, root.right)
    
    ################  DRIVER CODE  ######################
    root1 = TreeNode(70)
    root1.left = TreeNode(50)
    root1.right = TreeNode(90)
    root1.left.left = TreeNode(40)
    root1.left.right = TreeNode(60)
    root1.right.left = TreeNode(80)
    root1.right.right = TreeNode(95)
    root1.left.left.left = TreeNode(20)
    root1.left.right.left = TreeNode(55)
    root1.right.left.left = TreeNode(75)
    root1.right.left.right = TreeNode(85)
    root1.right.right.right = TreeNode(99)
    print(f"Number of nodes: {count_node(root)}") #9
    print(f"Height of the Root: {height(root)}") #3
    print(f"Number of levels: {get_level(root, 0, 9)}") #3
    print('-----IN ORDER-----') #8 4 9 2 5 1 6 3 7
    print_inorder(root)
    print()
    print('-----PRE ORDER-----') #1 8 4 9 2 5 6 3 7
    pre_order(root)
    print()
    print('-----POST ORDER-----')#8 4 9 2 5 6 3 7 1
    post_order(root)
    array=[None, "A", "B", "C", "D", "E", "F", "G", "H", None, None, None, "I", "J", None, "K"]
    print()
    print("Binary Tree from an array:") 
    tree=tree_contruction(array, 1)
    print_inorder(tree) #H D B E A I F J C G K
    print()
    array1=array_contruction(root1,1) #[None, 70, 50, 90, 40, 60, 80, 95, 20, None, 55, None, 75, 85, None, 99]
    print(array1)
    array2=array_contruction(root,1) #[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 55, None, 75, 85, None, 99]
    print(array2)
    print("Full Tree Checker:", is_full_tree(root)) #True
    if is_complete(root, 0, count_node(root)):
      print("complete Binary Tree") #tick
    else:
      print("Not a Complete Binary Tree")
    if is_perfect(root, depth(root)):
      print("perfect Binary Trtee")
    else:
      print("Not a Perfect Tree") #tick
    isSymmetric(root1)
          ''')

#-----------------------------------------------------------------------------


def heap():
    print('''
    #Heap Is a Data struture of a complete binary tree, where insertion is added at last and replacements is done with the last element and the top element then the top elements gets send to downwards as its max heap(parent bigger than child) and in mean heap parent is smaller than child
          ''')

#-----------------------------------------------------------------------------


def main():
    print('\nWelcome to Data Structure Notes'.upper())
    print('\nType:')
    print('A:Array, MA:multidimensional Array, SL:Singly linked list, DL: Doubly linked list') 
    print('S:Stack, Q:Queue, RE: Recursion, HA:Hashing, T:Tree, HE:Heap\n')
    inp=input('Enter the Code: ').upper()
    if inp=='A':
       array_list()
    if inp=='MA':
        multidimentional_array()
    if inp=='SL':
        singly_linked_list()
    if inp=='DL':
        doubly_linked_list()
    if inp=='S':
        stack()
    if inp=='Q':
        queue()
    if inp=='RE':
        recursion()
    if inp=='HA':
        hashing()
    if inp=='T':
        tree()
    if inp=='HE':
        heap()
    
    
main()