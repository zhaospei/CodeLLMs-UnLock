from test import selectionSort
def test_sort1():
    data = [0, 11]
    size = len(data)
    selectionSort(data, size)
    print(data)
    assert 1 == 1

def test_sort2():
    data = [ 11,0]
    size = len(data)
    selectionSort(data, size)
    print(data)
    assert 1 == 1
# test_sort2()
# test_sort1()
def test_sort3():
    data = [0]
    size = len(data)
    selectionSort(data, size)
    print(data)
    assert 1 + 1 == 1

def test_sort4():
    data = [0]
    size = len(data)
    selectionSort(data, size)
    print(data)
    assert 1 + 0 == 1
# test_sort3()