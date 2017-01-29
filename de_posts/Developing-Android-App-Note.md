---
title: Developing Android App Note
date: 2016-09-29 21:48:59
tags: App Developing
---

### Material Design + TextView

https://developers.google.com/products/

https://material.google.com/

https://developer.android.com/reference/android/widget/TextView.html

Android Studio:

 - https://developer.android.com/studio/write/layout-editor.html

Use the documenation as a resource to building your app screen. 



http://fragmentedpodcast.com/


Try `TextView`

```
<TextView
    android:text="Excited for the gift you'll surprise me with."
    android:background="#8E24AA"
    android:textColor="#F3E5F5"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content" 
    android:textSize="25sp"
    android:textStyle="bold" />
```

Try `ImageView`

```
<ImageView
    android:src="@drawable/ocean"
    android:layout_width="900dp"
    android:layout_height="900dp"
    android:scaleType="centerCrop"/>
```

## Lession 1B: Building Layouts

https://developer.android.com/reference/android/widget/TextView.html

```
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:orientation="vertical"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content">

    <TextView
        android:text="Guest List"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textSize="24sp"  />

    <TextView
        android:text="Kunal"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textSize="24sp"  />

</LinearLayout>
```


https://developer.android.com/guide/topics/ui/layout/linear.html?utm_source=udacity&utm_medium=course&utm_campaign=android_basics


RelativeLayout.LayoutParams

- https://developer.android.com/reference/android/widget/RelativeLayout.LayoutParams.html?utm_source=udacity&utm_medium=course&utm_campaign=android_basics


```
<RelativeLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <TextView
        android:id="@+id/lyla_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_alignParentLeft="true"
        android:textSize="24sp"
        android:text="Lyla" />

    <TextView
        android:id="@+id/me_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_toRightOf="@id/lyla_text_view"
        android:textSize="24sp"
        android:text="Me" />

    <TextView
        android:id="@+id/natalie_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textSize="24sp"
        android:text="Natalie" />

    <TextView
        android:id="@+id/jennie_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_alignParentRight="true"
        android:textSize="24sp"
        android:text="Jennie" />

    <TextView
        android:id="@+id/omoju_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentRight="true"
        android:layout_above="@id/jennie_text_view"
        android:textSize="24sp"
        android:text="Omoju" />

    <TextView
        android:id="@+id/amy_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_toLeftOf="@id/jennie_text_view"
        android:textSize="24sp"
        android:text="Amy" />

    <TextView
        android:id="@+id/ben_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_centerHorizontal="true"
        android:textSize="24sp"
        android:text="Ben" />

    <TextView
        android:id="@+id/kunal_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_above=
        android:textSize="24sp"
        android:text="Kunal" />

    <TextView
        android:id="@+id/kagure_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_toLeftOf="@id/ben_text_view"
        android:textSize="24sp"
        android:text="Kagure" />

</RelativeLayout>
```
